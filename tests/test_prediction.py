
"""
Tests unitaires pour le projet :
https://github.com/AlexandruEmil/Data-Science-API-FastAPI-Docker

Emplacement recommandÃ© dans le dÃ©pÃ´t clonÃ© :
    tests/unit/test_prediction.py

Objectif pÃ©dagogique :
- Tester uniquement la fonction predict(), sans lancer FastAPI.
- Couvrir les cas nominaux, limites, invalides et exceptionnels.
- Ã‰viter une couverture artificielle basÃ©e seulement sur des cas rÃ©pÃ©titifs.
"""

import math
import pytest

from app.utils import predict


# -----------------------------------------------------------------------------
# Cas nominaux : entrÃ©es valides et reprÃ©sentatives
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "features, expected",
    [
        ([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]),
        ([5.0], [10.0]),
        ([1.5, 2.5], [3.0, 5.0]),
    ],
)

def test_predict_nominal_cases(features, expected):
    """La fonction doit retourner une prÃ©diction conforme Ã  la rÃ¨gle y = 2x."""
    result = predict(features)
    assert result == pytest.approx(expected)



# -----------------------------------------------------------------------------
# Cas limites : valeurs particuliÃ¨res aux frontiÃ¨res du domaine
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "features, expected",
    [
        ([0.0], [0.0]),
        ([-1.0], [-2.0]),
        ([-1000.0], [-2000.0]),
        ([1_000_000.0], [2_000_000.0]),
        ([0.0, -1.0, 1_000_000.0], [0.0, -2.0, 2_000_000.0]),
    ],
)
def test_predict_boundary_cases(features, expected):
    """La fonction doit gÃ©rer correctement les valeurs limites ou particuliÃ¨res."""
    result = predict(features)

    assert result == pytest.approx(expected)


# -----------------------------------------------------------------------------
# Cas exceptionnel : entrÃ©e techniquement sous forme de liste, mais inexploitable
# -----------------------------------------------------------------------------

def test_predict_empty_list_raises_exception():
    """
    Une liste vide ne contient aucun Ã©chantillon Ã  prÃ©dire.
    Le modÃ¨le scikit-learn doit donc lever une exception.
    """
    with pytest.raises(ValueError):
        predict([])
        


# -----------------------------------------------------------------------------
# Cas invalides : donnÃ©es ne respectant pas les prÃ©conditions attendues
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "invalid_features",
    [
        None,
        "abc",
        {"feature1": 1.0},
        [1.0, "abc", 3.0],
        [1.0, None, 3.0],
    ],
)
def test_predict_invalid_inputs_raise_exception(invalid_features):
    """La fonction doit Ã©chouer de maniÃ¨re contrÃ´lÃ©e avec des entrÃ©es invalides."""
    with pytest.raises((TypeError, ValueError)):
        predict(invalid_features)



# -----------------------------------------------------------------------------
# PropriÃ©tÃ©s gÃ©nÃ©rales : tests plus robustes qu'une simple valeur attendue
# -----------------------------------------------------------------------------

def test_predict_output_is_a_list():
    """La fonction doit retourner une liste Python."""
    result = predict([1.0, 2.0, 3.0])

    assert isinstance(result, list)


def test_predict_output_size_matches_input_size():
    """Le nombre de prÃ©dictions doit correspondre au nombre d'entrÃ©es."""
    features = [1.0, 2.0, 3.0, 4.0]

    result = predict(features)

    assert len(result) == len(features)


def test_predict_output_values_are_numeric():
    """Chaque prÃ©diction doit Ãªtre une valeur numÃ©rique finie."""
    result = predict([1.0, 2.0, 3.0])

    assert all(isinstance(value, float) for value in result)
    assert all(math.isfinite(value) for value in result)


def test_predict_is_deterministic():
    """La fonction doit Ãªtre rejouable : mÃªmes entrÃ©es, mÃªmes sorties."""
    features = [3.5, 1.2, 4.9]

    first_result = predict(features)
    second_result = predict(features)

    assert first_result == pytest.approx(second_result)


def test_predict_does_not_modify_input_list():
    """La fonction ne doit pas modifier la liste reÃ§ue en entrÃ©e."""
    features = [1.0, 2.0, 3.0]
    original_features = features.copy()

    predict(features)

    assert features == original_features