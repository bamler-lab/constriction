import toml

t = toml.load("Cargo.toml")
crate_version = t['package']['version']

t = toml.load("pyproject.toml")
wheel_version = t['tool']['poetry']['version']

assert crate_version == wheel_version
print(crate_version)
