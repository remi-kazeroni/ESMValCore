import pytest

from esmvalcore.experimental.recipe_info import RecipeInfo
from esmvalcore.experimental.utils import (
    RecipeList,
    get_all_recipes,
    get_recipe,
)

pytest.importorskip(
    'esmvaltool',
    reason='The behaviour of these tests depends on what ``DIAGNOSTICS_PATH``'
    'points to. This is defined by a forward-reference to ESMValTool, which'
    'is not installed in the CI, but likely to be available in a developer'
    'or user installation.')


def test_get_recipe():
    """Get single recipe."""
    recipe = get_recipe('examples/recipe_python.yml')
    assert isinstance(recipe, RecipeInfo)


def test_get_all_recipes():
    """Get all recipes."""
    recipes = get_all_recipes()
    assert isinstance(recipes, list)

    recipes.find('')


def test_recipe_list_find():
    """Get all recipes."""
    recipes = get_all_recipes(subdir='examples')

    assert isinstance(recipes, RecipeList)

    result = recipes.find('python')

    assert isinstance(result, RecipeList)