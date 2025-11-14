# Final aggregated package re-exporting all phases
"""Final aggregated package re-exporting all phase APIs.

Assumes phase packages are installed and importable. Uses lazy imports to
avoid hard failures during partial installs.
"""

def _safe_import(mod_name):
	try:
		return __import__(mod_name, fromlist=['*'])
	except Exception:
		return None

_mods = [
	_safe_import('proyecto_bu'),
	_safe_import('proyecto_du'),
	_safe_import('proyecto_dp'),
	_safe_import('proyecto_modeling'),
	_safe_import('proyecto_eval'),
	_safe_import('proyecto_deploy'),
]

__all__ = []
for m in _mods:
	if m and hasattr(m, '__all__'):
		globals().update({name: getattr(m, name) for name in m.__all__})
		__all__.extend(m.__all__)
