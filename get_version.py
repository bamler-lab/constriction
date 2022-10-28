import toml

t = toml.load("Cargo.toml")
crate_version = t['package']['version']

print(crate_version)
