# Documentation Maintenance Instructions

This file contains instructions for keeping the README and documentation up to date with code changes.

## Documentation Structure

- `README.md` - Project overview, quick start, and basic examples
- `docs/features.md` - Detailed feature descriptions
- `docs/api-reference.md` - Complete API documentation
- `docs/cli-reference.md` - Command-line interface guide
- `docs/effects-guide.md` - In-depth effect documentation

## When to Update Documentation

### 1. Adding New Effects

When adding a new effect class to `src/distortion_effects.py`:

- **Update `docs/effects-guide.md`**:
  - Add a new section with effect name, description, and how it works
  - Document all parameters with value ranges and visual impact
  - Include command-line examples
  - Add creative applications and best practices

- **Update `docs/api-reference.md`**:
  - Add the effect class to the Effects section
  - Include constructor parameters and example code

- **Update `README.md`**:
  - Add the effect to the features list if it's a major addition
  - Add a quick example in the Quick Start section

### 2. Adding New Parameters

When adding parameters to existing effects:

- **Update `docs/effects-guide.md`**:
  - Add parameter to the relevant effect's Parameters section
  - Include value ranges and descriptions
  - Add examples showing the parameter in use

- **Update `docs/cli-reference.md`**:
  - Add new CLI flags if applicable
  - Update example commands

### 3. Adding New Features

When adding major features (e.g., new animation modes, rendering options):

- **Update `README.md`**:
  - Add to the Features section
  - Include a basic example

- **Update `docs/features.md`**:
  - Add detailed explanation
  - Include configuration examples
  - Document any limitations

### 4. Creating Example Scripts

When adding new example scripts to `examples/`:

- **Update `README.md`**:
  - Add to the Examples section with a brief description

- **Consider creating a corresponding effects guide section** if the example demonstrates a new technique

## Documentation Checklist

Before committing changes that affect functionality:

- [ ] Check if README.md needs updates
- [ ] Check if any docs/*.md files need updates
- [ ] Ensure all new parameters are documented
- [ ] Verify example code still works
- [ ] Update version numbers if applicable
- [ ] Add new examples if introducing complex features

## Current Documentation Gaps to Address

1. **Missing Features**:
   - CompositeEffect's `stretch_curves` parameter and `generate_factors_with_scale` method
   - Per-effect stretch curve types (constant, linear, ease_in, ease_out, ease_in_out)

2. **Missing Examples**:
   - Stacking effects demonstration
   - Ripple effect examples (if implemented)
   - Constant stretch curve usage

3. **Code-Documentation Sync**:
   - Ensure all effects in `distortion_effects.py` are documented
   - Verify all CLI parameters match documentation

## Style Guidelines

- Use clear, concise language
- Include code examples for all features
- Provide visual descriptions of effects
- Maintain consistent formatting across all documentation
- Test all example commands before documenting

## Regular Maintenance

- Review documentation quarterly for accuracy
- Update examples when dependencies change
- Check for broken links or outdated references
- Ensure new Python/CLI features are reflected in docs