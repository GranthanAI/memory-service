"""Production must not start with development-default or weak secrets; development stays unaffected."""

import pytest

from app.core.config import SystemSettings

STRONG = "k9Qx2mV7pLr4Zt8WyB3nHd6Fs1JcUe5A"
GOOD = dict(ENVIRONMENT="production", API_KEY=STRONG + "a", JWT_SECRET_KEY=STRONG + "b", USER_JWT_SECRET_KEY=STRONG + "c", DEBUG=False)
BAD = ["graphgpt-jwt-secret", "short", ""]


def build(**over):
    return SystemSettings(_env_file=None, **over)


def test_development_defaults_still_start():
    build()                                           # no ENVIRONMENT set: local development


def test_a_strong_production_configuration_starts():
    build(**GOOD)


@pytest.mark.parametrize("secret", BAD)
def test_production_refuses_default_weak_or_missing_jwt_secrets(secret):
    with pytest.raises(ValueError) as exc:
        build(**{**GOOD, "JWT_SECRET_KEY": secret})
    assert "JWT_SECRET_KEY" in str(exc.value)
    assert secret not in str(exc.value) or secret == ""            # the secret value is never echoed


def test_the_environment_name_is_case_insensitive():
    with pytest.raises(ValueError):
        build(**{**GOOD, "ENVIRONMENT": " Production ", "JWT_SECRET_KEY": "short"})


@pytest.mark.parametrize("field", ["API_KEY", "USER_JWT_SECRET_KEY"])
def test_production_requires_the_service_api_key_and_the_user_token_secret(field):
    with pytest.raises(ValueError) as exc:
        build(**{**GOOD, field: ""})
    assert field in str(exc.value)
