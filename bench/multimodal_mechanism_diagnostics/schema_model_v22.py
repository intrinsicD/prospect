"""Closed development schema primitives for future MM-008 v2.2 records.

This module is deliberately generic, in-memory contract infrastructure.  It
defines no MM-008 output schema, config, record, path, runtime policy, seed, or
formal authority.  The closed set of schema node types is sufficient to model
exact objects, sequences, tagged unions, and sibling-counted arrays without
accepting caller callbacks or subclasses as validators.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Final, NoReturn, TypeAlias, cast

PROTOCOL_SHA256: Final = "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
SCHEMA_VERSION: Final = "mm008-v2.2-development-schema-model-v1"
CLAIM_SCOPE: Final = "development-contract-infrastructure-only"
MAX_VALIDATION_DEPTH: Final = 128

Scalar: TypeAlias = None | bool | int | float | str


class SchemaDefinitionError(ValueError):
    """Raised when a schema definition is ambiguous, malformed, or open-ended."""


class SchemaValidationError(ValueError):
    """Raised when an in-memory value does not satisfy an issued schema tree."""


def _require_exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise SchemaDefinitionError(f"{label} must be an exact boolean")
    return cast(bool, value)


def _require_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or cast(int, value) < 0:
        raise SchemaDefinitionError(f"{label} must be a nonnegative exact integer")
    return cast(int, value)


def _require_name(value: object, label: str) -> str:
    if type(value) is not str or not cast(str, value):
        raise SchemaDefinitionError(f"{label} must be a nonempty exact string")
    return cast(str, value)


def _require_scalar(value: object, label: str) -> Scalar:
    if value is None or type(value) in {bool, int, str}:
        return cast(Scalar, value)
    if type(value) is float and math.isfinite(cast(float, value)):
        return cast(float, value)
    raise SchemaDefinitionError(f"{label} must be a finite built-in scalar")


def _bounded_int_label(value: int) -> str:
    """Return a stable label without invoking decimal conversion on huge ints."""

    bit_length = value.bit_length()
    if bit_length <= 256:
        return str(value)
    sign = "negative" if value < 0 else "nonnegative"
    return f"{sign} integer with {bit_length} bits"


def _bounded_string_label(value: str) -> str:
    if len(value) <= 64:
        return repr(value)
    return f"string with length {len(value)}"


def _bounded_scalar_label(value: Scalar) -> str:
    """Return a bounded diagnostic label for an exact built-in scalar."""

    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return _bounded_int_label(cast(int, value))
    if type(value) is float:
        return f"float {cast(float, value).hex()}"
    return _bounded_string_label(cast(str, value))


def _bounded_names_label(values: set[str] | list[str]) -> str:
    if len(values) > 4:
        return f"{len(values)} fields"
    return "[" + ", ".join(_bounded_string_label(value) for value in sorted(values)) + "]"


def _field_path(path: str, name: str) -> str:
    if len(name) <= 64 and name.isidentifier():
        return f"{path}.{name}"
    return f"{path}[field {_bounded_string_label(name)}]"


def _float_binary64_bits(value: float) -> bytes:
    return struct.pack(">d", value)


def _scalar_identity(value: Scalar) -> tuple[type[object], object]:
    value_type = cast(type[object], type(value))
    if type(value) is float:
        return value_type, _float_binary64_bits(cast(float, value))
    return value_type, value


def _scalar_matches(value: object, expected: Scalar) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is float:
        return _float_binary64_bits(cast(float, value)) == _float_binary64_bits(cast(float, expected))
    return value == expected


@dataclass(frozen=True, slots=True)
class NullSchema:
    """Accept only ``None``."""


@dataclass(frozen=True, slots=True)
class BooleanSchema:
    """Accept only an exact built-in boolean."""


@dataclass(frozen=True, slots=True)
class IntegerSchema:
    """Accept an exact built-in integer within optional inclusive bounds."""

    minimum: int | None = None
    maximum: int | None = None

    def __post_init__(self) -> None:
        if self.minimum is not None and type(self.minimum) is not int:
            raise SchemaDefinitionError("integer minimum must be an exact integer or null")
        if self.maximum is not None and type(self.maximum) is not int:
            raise SchemaDefinitionError("integer maximum must be an exact integer or null")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise SchemaDefinitionError("integer minimum exceeds maximum")


@dataclass(frozen=True, slots=True)
class FloatSchema:
    """Accept a finite built-in float, optionally also accepting exact integers."""

    minimum: float | None = None
    maximum: float | None = None
    allow_integer: bool = False

    def __post_init__(self) -> None:
        for value, label in ((self.minimum, "float minimum"), (self.maximum, "float maximum")):
            if value is not None and (type(value) is not float or not math.isfinite(value)):
                raise SchemaDefinitionError(f"{label} must be a finite exact float or null")
        _require_exact_bool(self.allow_integer, "float allow_integer")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise SchemaDefinitionError("float minimum exceeds maximum")


@dataclass(frozen=True, slots=True)
class StringSchema:
    """Accept an exact string within inclusive length and optional ASCII bounds."""

    minimum_length: int = 0
    maximum_length: int | None = None
    ascii_only: bool = False

    def __post_init__(self) -> None:
        minimum = _require_nonnegative_int(self.minimum_length, "string minimum length")
        if self.maximum_length is not None:
            maximum = _require_nonnegative_int(self.maximum_length, "string maximum length")
            if minimum > maximum:
                raise SchemaDefinitionError("string minimum length exceeds maximum length")
        _require_exact_bool(self.ascii_only, "string ascii_only")


@dataclass(frozen=True, slots=True)
class LiteralSchema:
    """Accept one exact built-in scalar, including its Python type."""

    value: Scalar

    def __post_init__(self) -> None:
        _require_scalar(self.value, "literal value")


@dataclass(frozen=True, slots=True)
class EnumSchema:
    """Accept one member of a finite exact built-in-scalar set."""

    values: tuple[Scalar, ...]

    def __post_init__(self) -> None:
        if type(self.values) is not tuple or not self.values:
            raise SchemaDefinitionError("enum values must be a nonempty exact tuple")
        checked = tuple(_require_scalar(value, "enum value") for value in self.values)
        identities = tuple(_scalar_identity(value) for value in checked)
        if len(set(identities)) != len(identities):
            raise SchemaDefinitionError("enum values contain a duplicate definition")


@dataclass(frozen=True, slots=True)
class ArraySchema:
    """Accept a homogeneous exact list within inclusive length bounds."""

    item: Schema
    minimum_length: int = 0
    maximum_length: int | None = None

    def __post_init__(self) -> None:
        _require_schema(self.item, "array item schema")
        if _unwrap_dependent_array(self.item) is not None:
            raise SchemaDefinitionError("dependent array must be a direct object field or one direct nullable wrapper")
        minimum = _require_nonnegative_int(self.minimum_length, "array minimum length")
        if self.maximum_length is not None:
            maximum = _require_nonnegative_int(self.maximum_length, "array maximum length")
            if minimum > maximum:
                raise SchemaDefinitionError("array minimum length exceeds maximum length")


@dataclass(frozen=True, slots=True)
class TupleSchema:
    """Accept an exact tuple with one schema per position."""

    items: tuple[Schema, ...]

    def __post_init__(self) -> None:
        if type(self.items) is not tuple:
            raise SchemaDefinitionError("tuple schema items must be an exact tuple")
        for index, item in enumerate(self.items):
            _require_schema(item, f"tuple schema item {index}")
            if _unwrap_dependent_array(item) is not None:
                raise SchemaDefinitionError(
                    "dependent array must be a direct object field or one direct nullable wrapper"
                )


@dataclass(frozen=True, slots=True)
class NullableSchema:
    """Explicitly accept either ``None`` or the wrapped schema."""

    inner: Schema

    def __post_init__(self) -> None:
        _require_schema(self.inner, "nullable inner schema")
        if type(self.inner) is NullSchema or type(self.inner) is NullableSchema:
            raise SchemaDefinitionError("nullable schema redundantly contains null")


@dataclass(frozen=True, slots=True)
class DependentArraySchema:
    """Accept a list whose exact length is derived from a sibling integer field."""

    item: Schema
    length_field: str
    multiplier: int = 1
    offset: int = 0
    maximum_length: int | None = None

    def __post_init__(self) -> None:
        _require_schema(self.item, "dependent-array item schema")
        if _unwrap_dependent_array(self.item) is not None:
            raise SchemaDefinitionError("dependent array must be a direct object field or one direct nullable wrapper")
        _require_name(self.length_field, "dependent-array length field")
        if type(self.multiplier) is not int or self.multiplier < 0:
            raise SchemaDefinitionError("dependent-array multiplier must be a nonnegative exact integer")
        if type(self.offset) is not int:
            raise SchemaDefinitionError("dependent-array offset must be an exact integer")
        if self.maximum_length is not None:
            _require_nonnegative_int(self.maximum_length, "dependent-array maximum length")


@dataclass(frozen=True, slots=True)
class FieldSchema:
    """One exact object field; absence is allowed only when ``required`` is false."""

    name: str
    schema: Schema
    required: bool = True

    def __post_init__(self) -> None:
        name = _require_name(self.name, "object field name")
        name_label = _bounded_string_label(name)
        _require_schema(self.schema, f"schema for field {name_label}")
        _require_exact_bool(self.required, f"required flag for field {name_label}")


@dataclass(frozen=True, slots=True)
class ObjectSchema:
    """Accept an exact string-keyed dict with no undeclared fields."""

    fields: tuple[FieldSchema, ...]

    def __post_init__(self) -> None:
        if type(self.fields) is not tuple:
            raise SchemaDefinitionError("object fields must be an exact tuple")
        if any(type(field) is not FieldSchema for field in self.fields):
            raise SchemaDefinitionError("object fields contain a non-FieldSchema definition")
        names = tuple(field.name for field in self.fields)
        if len(set(names)) != len(names):
            raise SchemaDefinitionError("object fields contain a duplicate definition")
        by_name = {field.name: field for field in self.fields}
        for field in self.fields:
            dependent = _unwrap_dependent_array(field.schema)
            if dependent is None:
                continue
            if dependent.length_field == field.name:
                raise SchemaDefinitionError("dependent array cannot use its own field as its length")
            source = by_name.get(dependent.length_field)
            if source is None:
                raise SchemaDefinitionError("dependent array references an undeclared length field")
            if not source.required or type(source.schema) is not IntegerSchema:
                raise SchemaDefinitionError("dependent-array length field must be a required exact integer field")


@dataclass(frozen=True, slots=True)
class UnionVariant:
    """One discriminator literal and its exact object schema."""

    tag: str
    schema: ObjectSchema

    def __post_init__(self) -> None:
        _require_name(self.tag, "union variant tag")
        if type(self.schema) is not ObjectSchema:
            raise SchemaDefinitionError("union variant schema must be an exact ObjectSchema")


@dataclass(frozen=True, slots=True)
class DiscriminatedUnionSchema:
    """Dispatch an exact object through one required string discriminator field."""

    discriminator: str
    variants: tuple[UnionVariant, ...]

    def __post_init__(self) -> None:
        discriminator = _require_name(self.discriminator, "union discriminator")
        if type(self.variants) is not tuple or not self.variants:
            raise SchemaDefinitionError("union variants must be a nonempty exact tuple")
        if any(type(variant) is not UnionVariant for variant in self.variants):
            raise SchemaDefinitionError("union variants contain a non-UnionVariant definition")
        tags = tuple(variant.tag for variant in self.variants)
        if len(set(tags)) != len(tags):
            raise SchemaDefinitionError("union variants contain a duplicate discriminator")
        for variant in self.variants:
            field = next(
                (item for item in variant.schema.fields if item.name == discriminator),
                None,
            )
            if (
                field is None
                or not field.required
                or type(field.schema) is not LiteralSchema
                or type(field.schema.value) is not str
                or field.schema.value != variant.tag
            ):
                raise SchemaDefinitionError("union variant must require a matching string LiteralSchema discriminator")


Schema: TypeAlias = (
    NullSchema
    | BooleanSchema
    | IntegerSchema
    | FloatSchema
    | StringSchema
    | LiteralSchema
    | EnumSchema
    | ArraySchema
    | TupleSchema
    | NullableSchema
    | DependentArraySchema
    | ObjectSchema
    | DiscriminatedUnionSchema
)

_SCHEMA_TYPES: Final = (
    NullSchema,
    BooleanSchema,
    IntegerSchema,
    FloatSchema,
    StringSchema,
    LiteralSchema,
    EnumSchema,
    ArraySchema,
    TupleSchema,
    NullableSchema,
    DependentArraySchema,
    ObjectSchema,
    DiscriminatedUnionSchema,
)


def _require_schema(value: object, label: str) -> Schema:
    if type(value) not in _SCHEMA_TYPES:
        raise SchemaDefinitionError(f"{label} is not in the closed schema-node set")
    return cast(Schema, value)


def _unwrap_dependent_array(schema: Schema) -> DependentArraySchema | None:
    if type(schema) is DependentArraySchema:
        return cast(DependentArraySchema, schema)
    if type(schema) is NullableSchema and type(schema.inner) is DependentArraySchema:
        return cast(DependentArraySchema, schema.inner)
    return None


def _validate_schema_integrity(
    schema: Schema,
    *,
    placement: str,
    depth: int,
    ancestors: set[int],
) -> None:
    """Check graph safety and context-sensitive nodes before value validation."""

    if depth > MAX_VALIDATION_DEPTH:
        raise SchemaDefinitionError(f"schema nesting exceeds {MAX_VALIDATION_DEPTH}")
    identity = id(schema)
    if identity in ancestors:
        raise SchemaDefinitionError("schema graph contains a cycle")
    ancestors.add(identity)
    try:
        schema_type = type(schema)
        if schema_type in {
            NullSchema,
            BooleanSchema,
            IntegerSchema,
            FloatSchema,
            StringSchema,
            LiteralSchema,
            EnumSchema,
        }:
            return
        if schema_type is ArraySchema:
            array = cast(ArraySchema, schema)
            _validate_schema_integrity(
                array.item,
                placement="nested",
                depth=depth + 1,
                ancestors=ancestors,
            )
            return
        if schema_type is TupleSchema:
            positional = cast(TupleSchema, schema)
            for item in positional.items:
                _validate_schema_integrity(
                    item,
                    placement="nested",
                    depth=depth + 1,
                    ancestors=ancestors,
                )
            return
        if schema_type is NullableSchema:
            nullable = cast(NullableSchema, schema)
            inner_placement = (
                "nullable-object-field"
                if placement == "object-field" and type(nullable.inner) is DependentArraySchema
                else "nested"
            )
            _validate_schema_integrity(
                nullable.inner,
                placement=inner_placement,
                depth=depth + 1,
                ancestors=ancestors,
            )
            return
        if schema_type is DependentArraySchema:
            if placement not in {"object-field", "nullable-object-field"}:
                raise SchemaDefinitionError(
                    "dependent array must be a direct object field or one direct nullable wrapper"
                )
            dependent = cast(DependentArraySchema, schema)
            _validate_schema_integrity(
                dependent.item,
                placement="nested",
                depth=depth + 1,
                ancestors=ancestors,
            )
            return
        if schema_type is ObjectSchema:
            exact = cast(ObjectSchema, schema)
            for field in exact.fields:
                _validate_schema_integrity(
                    field.schema,
                    placement="object-field",
                    depth=depth + 1,
                    ancestors=ancestors,
                )
            return
        if schema_type is DiscriminatedUnionSchema:
            union = cast(DiscriminatedUnionSchema, schema)
            for variant in union.variants:
                _validate_schema_integrity(
                    variant.schema,
                    placement="nested",
                    depth=depth + 1,
                    ancestors=ancestors,
                )
            return
        raise SchemaDefinitionError("schema graph contains a node outside the closed set")
    finally:
        ancestors.remove(identity)


def _fail(path: str, message: str) -> NoReturn:
    raise SchemaValidationError(f"{path}: {message}")


def _validate_scalar_identity(value: object, expected: Scalar, path: str) -> None:
    if not _scalar_matches(value, expected):
        _fail(path, f"expected exact literal {_bounded_scalar_label(expected)}")


def _validate(
    schema: Schema,
    value: object,
    *,
    path: str,
    parent: dict[str, object] | None,
    depth: int,
) -> None:
    if depth > MAX_VALIDATION_DEPTH:
        _fail(path, f"validation nesting exceeds {MAX_VALIDATION_DEPTH}")
    schema_type = type(schema)
    if schema_type is NullSchema:
        if value is not None:
            _fail(path, "expected null")
        return
    if schema_type is BooleanSchema:
        if type(value) is not bool:
            _fail(path, "expected an exact boolean")
        return
    if schema_type is IntegerSchema:
        integer = cast(IntegerSchema, schema)
        if type(value) is not int:
            _fail(path, "expected an exact integer")
        checked = cast(int, value)
        if integer.minimum is not None and checked < integer.minimum:
            _fail(
                path,
                f"integer is below minimum {_bounded_int_label(integer.minimum)}",
            )
        if integer.maximum is not None and checked > integer.maximum:
            _fail(
                path,
                f"integer is above maximum {_bounded_int_label(integer.maximum)}",
            )
        return
    if schema_type is FloatSchema:
        floating = cast(FloatSchema, schema)
        if type(value) is float:
            checked_number: int | float = cast(float, value)
        elif floating.allow_integer and type(value) is int:
            checked_number = cast(int, value)
        else:
            _fail(path, "expected a finite exact float")
        if type(checked_number) is float and not math.isfinite(checked_number):
            _fail(path, "float is nonfinite")
        if floating.minimum is not None and checked_number < floating.minimum:
            _fail(path, f"float is below minimum {floating.minimum}")
        if floating.maximum is not None and checked_number > floating.maximum:
            _fail(path, f"float is above maximum {floating.maximum}")
        return
    if schema_type is StringSchema:
        string = cast(StringSchema, schema)
        if type(value) is not str:
            _fail(path, "expected an exact string")
        checked_string = cast(str, value)
        if len(checked_string) < string.minimum_length:
            _fail(
                path,
                f"string is shorter than {_bounded_int_label(string.minimum_length)}",
            )
        if string.maximum_length is not None and len(checked_string) > string.maximum_length:
            _fail(
                path,
                f"string is longer than {_bounded_int_label(string.maximum_length)}",
            )
        if string.ascii_only and not checked_string.isascii():
            _fail(path, "string is not ASCII")
        return
    if schema_type is LiteralSchema:
        _validate_scalar_identity(value, cast(LiteralSchema, schema).value, path)
        return
    if schema_type is EnumSchema:
        choices = cast(EnumSchema, schema).values
        if not any(_scalar_matches(value, choice) for choice in choices):
            _fail(path, "value is not an exact enum member")
        return
    if schema_type is ArraySchema:
        array = cast(ArraySchema, schema)
        if type(value) is not list:
            _fail(path, "expected an exact list")
        array_items = cast(list[object], value)
        if len(array_items) < array.minimum_length:
            _fail(
                path,
                f"list is shorter than {_bounded_int_label(array.minimum_length)}",
            )
        if array.maximum_length is not None and len(array_items) > array.maximum_length:
            _fail(
                path,
                f"list is longer than {_bounded_int_label(array.maximum_length)}",
            )
        for index, item in enumerate(array_items):
            _validate(array.item, item, path=f"{path}[{index}]", parent=None, depth=depth + 1)
        return
    if schema_type is TupleSchema:
        positional = cast(TupleSchema, schema)
        if type(value) is not tuple:
            _fail(path, "expected an exact tuple")
        tuple_items = cast(tuple[object, ...], value)
        if len(tuple_items) != len(positional.items):
            _fail(path, f"tuple length differs from exact length {len(positional.items)}")
        for index, (item_schema, item) in enumerate(zip(positional.items, tuple_items, strict=True)):
            _validate(item_schema, item, path=f"{path}[{index}]", parent=None, depth=depth + 1)
        return
    if schema_type is NullableSchema:
        if value is None:
            return
        nullable = cast(NullableSchema, schema)
        inner_parent = parent if type(nullable.inner) is DependentArraySchema else None
        _validate(nullable.inner, value, path=path, parent=inner_parent, depth=depth + 1)
        return
    if schema_type is DependentArraySchema:
        dependent = cast(DependentArraySchema, schema)
        if parent is None:
            _fail(path, "dependent array has no containing object")
        if type(value) is not list:
            _fail(path, "expected an exact dependent list")
        count_value = parent.get(dependent.length_field)
        if type(count_value) is not int:
            _fail(path, "dependent length source is not an exact integer")
        expected = cast(int, count_value) * dependent.multiplier + dependent.offset
        if expected < 0:
            _fail(path, "dependent length expression is negative")
        if dependent.maximum_length is not None and expected > dependent.maximum_length:
            _fail(path, "dependent length exceeds its declared maximum")
        dependent_items = cast(list[object], value)
        if len(dependent_items) != expected:
            _fail(
                path,
                f"list length {len(dependent_items)} differs from dependent length {_bounded_int_label(expected)}",
            )
        for index, item in enumerate(dependent_items):
            _validate(
                dependent.item,
                item,
                path=f"{path}[{index}]",
                parent=None,
                depth=depth + 1,
            )
        return
    if schema_type is ObjectSchema:
        exact = cast(ObjectSchema, schema)
        if type(value) is not dict:
            _fail(path, "expected an exact object")
        raw = cast(dict[object, object], value)
        if any(type(key) is not str for key in raw):
            _fail(path, "object contains a non-string key")
        checked_object = cast(dict[str, object], raw)
        declared = {field.name for field in exact.fields}
        extras = set(checked_object) - declared
        if extras:
            _fail(path, f"object contains undeclared fields {_bounded_names_label(extras)}")
        missing = [field.name for field in exact.fields if field.required and field.name not in checked_object]
        if missing:
            _fail(path, f"object is missing required fields {_bounded_names_label(missing)}")
        for field in exact.fields:
            if field.name in checked_object:
                _validate(
                    field.schema,
                    checked_object[field.name],
                    path=_field_path(path, field.name),
                    parent=checked_object,
                    depth=depth + 1,
                )
        return
    if schema_type is DiscriminatedUnionSchema:
        union = cast(DiscriminatedUnionSchema, schema)
        if type(value) is not dict:
            _fail(path, "expected an exact discriminated object")
        raw_union = cast(dict[object, object], value)
        if any(type(key) is not str for key in raw_union):
            _fail(path, "discriminated object contains a non-string key")
        object_value = cast(dict[str, object], raw_union)
        if union.discriminator not in object_value:
            _fail(
                path,
                f"discriminated object lacks {_bounded_string_label(union.discriminator)}",
            )
        tag = object_value[union.discriminator]
        if type(tag) is not str:
            _fail(path, "union discriminator is not an exact string")
        variant = next((item for item in union.variants if item.tag == tag), None)
        if variant is None:
            _fail(
                path,
                f"unknown union discriminator {_bounded_string_label(cast(str, tag))}",
            )
        _validate(variant.schema, object_value, path=path, parent=None, depth=depth + 1)
        return
    raise SchemaDefinitionError("validator received a schema outside the closed node set")


def validate(schema: Schema, value: object) -> None:
    """Validate one in-memory value or raise a stable contract exception."""

    checked = _require_schema(schema, "root schema")
    try:
        _validate_schema_integrity(
            checked,
            placement="root",
            depth=0,
            ancestors=set(),
        )
    except RecursionError as error:
        raise SchemaDefinitionError("schema nesting exceeds the interpreter limit") from error
    try:
        _validate(checked, value, path="$", parent=None, depth=0)
    except RecursionError as error:
        raise SchemaValidationError("$: validation nesting exceeds the interpreter limit") from error


__all__ = [
    "CLAIM_SCOPE",
    "MAX_VALIDATION_DEPTH",
    "PROTOCOL_SHA256",
    "SCHEMA_VERSION",
    "ArraySchema",
    "BooleanSchema",
    "DependentArraySchema",
    "DiscriminatedUnionSchema",
    "EnumSchema",
    "FieldSchema",
    "FloatSchema",
    "IntegerSchema",
    "LiteralSchema",
    "NullSchema",
    "NullableSchema",
    "ObjectSchema",
    "Scalar",
    "Schema",
    "SchemaDefinitionError",
    "SchemaValidationError",
    "StringSchema",
    "TupleSchema",
    "UnionVariant",
    "validate",
]
