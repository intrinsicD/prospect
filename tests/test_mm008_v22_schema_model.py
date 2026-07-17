"""Adversarial tests for the closed MM-008 v2.2 development schema model."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from bench.multimodal_mechanism_diagnostics import schema_model_v22 as model


def _kind_field(tag: str) -> model.FieldSchema:
    return model.FieldSchema("kind", model.LiteralSchema(tag))


def test_identity_is_explicitly_development_only_and_contains_no_authority_api() -> None:
    assert model.PROTOCOL_SHA256 == ("300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622")
    assert model.SCHEMA_VERSION == "mm008-v2.2-development-schema-model-v1"
    assert model.CLAIM_SCOPE == "development-contract-infrastructure-only"
    assert model.__all__ == [
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


def test_exact_object_accepts_only_declared_fields_and_required_presence() -> None:
    schema = model.ObjectSchema(
        (
            model.FieldSchema("name", model.StringSchema(1, 8, ascii_only=True)),
            model.FieldSchema("count", model.IntegerSchema(0, 4)),
            model.FieldSchema("note", model.StringSchema(maximum_length=12), required=False),
        )
    )
    model.validate(schema, {"name": "alpha", "count": 2})
    model.validate(schema, {"name": "alpha", "count": 2, "note": "ok"})

    with pytest.raises(model.SchemaValidationError, match="missing required.*count"):
        model.validate(schema, {"name": "alpha"})
    with pytest.raises(model.SchemaValidationError, match="undeclared.*extra"):
        model.validate(schema, {"name": "alpha", "count": 2, "extra": 1})
    with pytest.raises(model.SchemaValidationError, match="non-string key"):
        model.validate(schema, {"name": "alpha", "count": 2, 3: "invalid"})
    with pytest.raises(model.SchemaValidationError, match="expected an exact object"):
        model.validate(schema, (("name", "alpha"), ("count", 2)))


def test_nullable_and_optional_are_independent_explicit_contracts() -> None:
    schema = model.ObjectSchema(
        (
            model.FieldSchema("required_nullable", model.NullableSchema(model.IntegerSchema(0, 3))),
            model.FieldSchema("optional_nonnull", model.IntegerSchema(0, 3), required=False),
        )
    )
    model.validate(schema, {"required_nullable": None})
    model.validate(schema, {"required_nullable": 1})
    model.validate(schema, {"required_nullable": 1, "optional_nonnull": 2})

    with pytest.raises(model.SchemaValidationError, match="missing required.*required_nullable"):
        model.validate(schema, {})
    with pytest.raises(model.SchemaValidationError, match="optional_nonnull.*exact integer"):
        model.validate(schema, {"required_nullable": 1, "optional_nonnull": None})


def test_arrays_and_tuples_have_distinct_exact_container_contracts() -> None:
    array = model.ArraySchema(model.IntegerSchema(0, 3), minimum_length=1, maximum_length=3)
    positional = model.TupleSchema((model.StringSchema(1, 4), model.BooleanSchema()))
    model.validate(array, [0, 2, 3])
    model.validate(positional, ("ok", True))

    with pytest.raises(model.SchemaValidationError, match="exact list"):
        model.validate(array, (0, 2))
    with pytest.raises(model.SchemaValidationError, match="longer than 3"):
        model.validate(array, [0, 1, 2, 3])
    with pytest.raises(model.SchemaValidationError, match="exact tuple"):
        model.validate(positional, ["ok", True])
    with pytest.raises(model.SchemaValidationError, match="exact length 2"):
        model.validate(positional, ("ok",))


def test_literals_and_enums_compare_both_value_and_exact_type() -> None:
    model.validate(model.LiteralSchema(True), True)
    model.validate(model.EnumSchema((1, "one", False, None)), "one")
    model.validate(model.EnumSchema((1, "one", False, None)), 1)

    with pytest.raises(model.SchemaValidationError, match="exact literal"):
        model.validate(model.LiteralSchema(True), 1)
    with pytest.raises(model.SchemaValidationError, match="enum member"):
        model.validate(model.EnumSchema((1, "one", False, None)), True)


def test_float_literals_and_enums_use_exact_binary64_identity() -> None:
    negative_zero = model.LiteralSchema(-0.0)
    positive_zero = model.LiteralSchema(0.0)
    model.validate(negative_zero, -0.0)
    model.validate(positive_zero, 0.0)

    with pytest.raises(model.SchemaValidationError, match="exact literal"):
        model.validate(negative_zero, 0.0)
    with pytest.raises(model.SchemaValidationError, match="exact literal"):
        model.validate(positive_zero, -0.0)

    both_zeros = model.EnumSchema((-0.0, 0.0))
    model.validate(both_zeros, -0.0)
    model.validate(both_zeros, 0.0)
    with pytest.raises(model.SchemaDefinitionError, match="duplicate definition"):
        model.EnumSchema((-0.0, -0.0))


def test_bounded_scalars_reject_bool_as_int_and_all_nonfinite_floats() -> None:
    integer = model.IntegerSchema(0, 5)
    floating = model.FloatSchema(-1.0, 1.0)
    numeric_float = model.FloatSchema(-1.0, 1.0, allow_integer=True)
    ascii_text = model.StringSchema(1, 3, ascii_only=True)
    model.validate(integer, 5)
    model.validate(floating, -0.0)
    model.validate(numeric_float, 1)
    model.validate(ascii_text, "abc")

    with pytest.raises(model.SchemaValidationError, match="exact integer"):
        model.validate(integer, True)
    with pytest.raises(model.SchemaValidationError, match="exact float"):
        model.validate(floating, 1)
    with pytest.raises(model.SchemaValidationError, match="exact float"):
        model.validate(numeric_float, False)
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(model.SchemaValidationError, match="nonfinite"):
            model.validate(floating, value)
    with pytest.raises(model.SchemaValidationError, match="above maximum"):
        model.validate(integer, 6)
    with pytest.raises(model.SchemaValidationError, match="float is above maximum"):
        model.validate(numeric_float, 10**1_000)
    with pytest.raises(model.SchemaValidationError, match="not ASCII"):
        model.validate(ascii_text, "é")


def test_validation_diagnostics_are_bounded_for_5000_digit_integer_inputs() -> None:
    huge = 10**4_999
    cases: tuple[tuple[model.Schema, object], ...] = (
        (model.IntegerSchema(minimum=huge), 0),
        (model.IntegerSchema(maximum=-huge), 0),
        (model.LiteralSchema(huge), 0),
        (model.StringSchema(minimum_length=huge), ""),
        (model.ArraySchema(model.IntegerSchema(), minimum_length=huge), []),
    )
    for schema, value in cases:
        with pytest.raises(model.SchemaValidationError) as captured:
            model.validate(schema, value)
        message = str(captured.value)
        assert "bits" in message
        assert len(message) < 256

    dependent = model.ObjectSchema(
        (
            model.FieldSchema("count", model.IntegerSchema()),
            model.FieldSchema(
                "items",
                model.DependentArraySchema(
                    model.IntegerSchema(),
                    "count",
                    multiplier=huge,
                    offset=huge,
                ),
            ),
        )
    )
    with pytest.raises(model.SchemaValidationError) as captured:
        model.validate(dependent, {"count": huge, "items": []})
    message = str(captured.value)
    assert "dependent length" in message
    assert "bits" in message
    assert len(message) < 256

    capped = model.ObjectSchema(
        (
            model.FieldSchema("count", model.IntegerSchema()),
            model.FieldSchema(
                "items",
                model.DependentArraySchema(
                    model.IntegerSchema(),
                    "count",
                    multiplier=2,
                    maximum_length=huge,
                ),
            ),
        )
    )
    with pytest.raises(model.SchemaValidationError, match="declared maximum") as captured:
        model.validate(capped, {"count": huge, "items": []})
    assert len(str(captured.value)) < 256


def test_dependent_array_uses_required_sibling_integer_and_exact_expression() -> None:
    schema = model.ObjectSchema(
        (
            model.FieldSchema("count", model.IntegerSchema(0, 5)),
            model.FieldSchema(
                "items",
                model.DependentArraySchema(
                    model.StringSchema(1, 4),
                    "count",
                    multiplier=2,
                    offset=1,
                    maximum_length=9,
                ),
            ),
        )
    )
    model.validate(schema, {"count": 2, "items": ["a", "b", "c", "d", "e"]})

    with pytest.raises(model.SchemaValidationError, match="dependent length 5"):
        model.validate(schema, {"count": 2, "items": ["a", "b", "c", "d"]})
    with pytest.raises(model.SchemaValidationError, match="count.*exact integer"):
        model.validate(schema, {"count": True, "items": ["a", "b", "c"]})
    with pytest.raises(model.SchemaValidationError, match="declared maximum"):
        model.validate(schema, {"count": 5, "items": []})


def test_dependent_array_is_only_legal_as_an_object_field_or_one_nullable_wrapper() -> None:
    dependent = model.DependentArraySchema(model.IntegerSchema(), "count")

    with pytest.raises(model.SchemaDefinitionError, match="direct object field"):
        model.validate(dependent, [])
    with pytest.raises(model.SchemaDefinitionError, match="direct object field"):
        model.validate(model.NullableSchema(dependent), None)
    with pytest.raises(model.SchemaDefinitionError, match="direct object field"):
        model.ArraySchema(dependent)
    with pytest.raises(model.SchemaDefinitionError, match="direct object field"):
        model.TupleSchema((model.NullableSchema(dependent),))
    with pytest.raises(model.SchemaDefinitionError, match="direct object field"):
        model.DependentArraySchema(model.NullableSchema(dependent), "count")

    nullable_field = model.ObjectSchema(
        (
            model.FieldSchema("count", model.IntegerSchema(0, 1)),
            model.FieldSchema("items", model.NullableSchema(dependent)),
        )
    )
    model.validate(nullable_field, {"count": 0, "items": None})
    model.validate(nullable_field, {"count": 1, "items": [3]})


def test_nested_object_dependent_length_uses_only_its_nearest_object() -> None:
    child = model.ObjectSchema(
        (
            model.FieldSchema("count", model.IntegerSchema(0, 2)),
            model.FieldSchema(
                "items",
                model.DependentArraySchema(model.IntegerSchema(), "count"),
            ),
        )
    )
    parent = model.ObjectSchema(
        (
            model.FieldSchema("count", model.IntegerSchema(99, 99)),
            model.FieldSchema("children", model.ArraySchema(child, 1, 1)),
        )
    )
    model.validate(
        parent,
        {"count": 99, "children": [{"count": 1, "items": [7]}]},
    )

    with pytest.raises(model.SchemaValidationError, match="dependent length 2"):
        model.validate(
            parent,
            {"count": 99, "children": [{"count": 2, "items": [7]}]},
        )


def test_discriminated_union_dispatches_only_matching_exact_object_variant() -> None:
    alpha = model.ObjectSchema(
        (
            _kind_field("alpha"),
            model.FieldSchema("value", model.IntegerSchema(0, 3)),
        )
    )
    beta = model.ObjectSchema(
        (
            _kind_field("beta"),
            model.FieldSchema("value", model.StringSchema(1, 5)),
        )
    )
    schema = model.DiscriminatedUnionSchema(
        "kind",
        (model.UnionVariant("alpha", alpha), model.UnionVariant("beta", beta)),
    )
    model.validate(schema, {"kind": "alpha", "value": 2})
    model.validate(schema, {"kind": "beta", "value": "two"})

    with pytest.raises(model.SchemaValidationError, match="lacks 'kind'"):
        model.validate(schema, {"value": 2})
    with pytest.raises(model.SchemaValidationError, match="not an exact string"):
        model.validate(schema, {"kind": 1, "value": 2})
    with pytest.raises(model.SchemaValidationError, match="unknown union discriminator"):
        model.validate(schema, {"kind": "gamma", "value": 2})
    with pytest.raises(model.SchemaValidationError, match="value.*exact integer"):
        model.validate(schema, {"kind": "alpha", "value": "two"})
    with pytest.raises(model.SchemaValidationError, match="undeclared.*extra"):
        model.validate(schema, {"kind": "alpha", "value": 2, "extra": False})


def test_duplicate_or_malformed_schema_definitions_fail_closed() -> None:
    with pytest.raises(model.SchemaDefinitionError, match="duplicate definition"):
        model.ObjectSchema(
            (
                model.FieldSchema("value", model.IntegerSchema()),
                model.FieldSchema("value", model.StringSchema()),
            )
        )
    with pytest.raises(model.SchemaDefinitionError, match="duplicate definition"):
        model.EnumSchema(("a", "a"))

    alpha = model.ObjectSchema((_kind_field("alpha"),))
    with pytest.raises(model.SchemaDefinitionError, match="duplicate discriminator"):
        model.DiscriminatedUnionSchema(
            "kind",
            (model.UnionVariant("alpha", alpha), model.UnionVariant("alpha", alpha)),
        )
    with pytest.raises(model.SchemaDefinitionError, match="matching string LiteralSchema"):
        model.DiscriminatedUnionSchema(
            "kind",
            (
                model.UnionVariant(
                    "alpha",
                    model.ObjectSchema((model.FieldSchema("kind", model.EnumSchema(("alpha", "beta"))),)),
                ),
            ),
        )


def test_dependent_schema_definition_rejects_missing_optional_or_wrong_count_fields() -> None:
    dependent = model.DependentArraySchema(model.IntegerSchema(), "count")
    with pytest.raises(model.SchemaDefinitionError, match="undeclared length field"):
        model.ObjectSchema((model.FieldSchema("items", dependent),))
    with pytest.raises(model.SchemaDefinitionError, match="required exact integer"):
        model.ObjectSchema(
            (
                model.FieldSchema("count", model.IntegerSchema(), required=False),
                model.FieldSchema("items", dependent),
            )
        )
    with pytest.raises(model.SchemaDefinitionError, match="required exact integer"):
        model.ObjectSchema(
            (
                model.FieldSchema("count", model.StringSchema()),
                model.FieldSchema("items", dependent),
            )
        )
    with pytest.raises(model.SchemaDefinitionError, match="own field"):
        model.ObjectSchema(
            (
                model.FieldSchema(
                    "items",
                    model.DependentArraySchema(model.IntegerSchema(), "items"),
                ),
            )
        )


def test_schema_definitions_reject_nonfinite_bounds_literals_and_redundant_nullability() -> None:
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(model.SchemaDefinitionError, match="finite exact float"):
            model.FloatSchema(minimum=value)
        with pytest.raises(model.SchemaDefinitionError, match="finite built-in scalar"):
            model.LiteralSchema(value)
        with pytest.raises(model.SchemaDefinitionError, match="finite built-in scalar"):
            model.EnumSchema((value,))
    with pytest.raises(model.SchemaDefinitionError, match="minimum exceeds maximum"):
        model.IntegerSchema(2, 1)
    with pytest.raises(model.SchemaDefinitionError, match="redundantly contains null"):
        model.NullableSchema(model.NullSchema())
    with pytest.raises(model.SchemaDefinitionError, match="redundantly contains null"):
        model.NullableSchema(model.NullableSchema(model.IntegerSchema()))


def test_closed_node_set_rejects_arbitrary_callback_or_lookalike_schema() -> None:
    class CallbackSchema:
        def validate(self, value: object) -> bool:
            del value
            return True

    with pytest.raises(model.SchemaDefinitionError, match="closed schema-node set"):
        model.ArraySchema(CallbackSchema())  # type: ignore[arg-type]
    with pytest.raises(model.SchemaDefinitionError, match="closed schema-node set"):
        model.validate(CallbackSchema(), "anything")  # type: ignore[arg-type]


def test_public_validation_rechecks_forged_placement_and_schema_cycles() -> None:
    dependent = model.DependentArraySchema(model.IntegerSchema(), "count")
    forged = model.ArraySchema(model.IntegerSchema())
    object.__setattr__(forged, "item", dependent)
    with pytest.raises(model.SchemaDefinitionError, match="direct object field"):
        model.validate(forged, [])

    cyclic = model.ArraySchema(model.IntegerSchema())
    object.__setattr__(cyclic, "item", cyclic)
    with pytest.raises(model.SchemaDefinitionError, match="schema graph contains a cycle"):
        model.validate(cyclic, [])


def test_schema_integrity_depth_is_bounded_with_a_stable_contract_error() -> None:
    schema: model.Schema = model.IntegerSchema()
    value: object = 1
    for _ in range(model.MAX_VALIDATION_DEPTH + 2):
        schema = model.ArraySchema(schema, minimum_length=1, maximum_length=1)
        value = [value]
    with pytest.raises(model.SchemaDefinitionError, match="schema nesting exceeds"):
        model.validate(schema, value)


def test_schema_nodes_are_immutable() -> None:
    schema = model.IntegerSchema(0, 1)
    with pytest.raises(FrozenInstanceError):
        schema.maximum = 2  # type: ignore[misc]
