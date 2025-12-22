# Bug Report and Fixes

## Summary

Found and fixed 6 bugs in the codebase affecting type safety, imports, and runtime behavior.

---

## Bugs Found and Fixed

### 1. **Missing Import in spectroscopy.py**

- **File**: [echospec/simulation/spectroscopy.py](echospec/simulation/spectroscopy.py)
- **Line**: 120
- **Issue**: `PulseType` is used in `__main__` section but not imported
- **Error**: `"PulseType" is not defined`
- **Fix**: Added `from echospec.simulation.pulses import PulseType`

---

### 2. **Nullable pulse_type Parameter in Hamiltonian**

- **File**: [echospec/simulation/hamiltonian.py](echospec/simulation/hamiltonian.py)
- **Line**: 15-16
- **Issue**: `self.params.pulse_type` can be `None` but is passed to `choose_pulse()` which expects `PulseType`
- **Error**: `Argument of type "PulseType | None" cannot be assigned to parameter "pulse_type" of type "PulseType"`
- **Fix**: Added validation to check if `pulse_type` is `None` and raise a descriptive error

---

### 3. **Uncallable None Pulse Function**

- **File**: [echospec/simulation/hamiltonian.py](echospec/simulation/hamiltonian.py)
- **Line**: 37
- **Issue**: `_pulse` can be `None` for `SQUARE` pulse type but is called without checking
- **Error**: `Object of type "None" cannot be called`
- **Fix**: Added check: if `_pulse is None`, return `np.ones_like(t)`

---

### 4. **Missing numpy Import in Hamiltonian**

- **File**: [echospec/simulation/hamiltonian.py](echospec/simulation/hamiltonian.py)
- **Line**: 38 (after fix)
- **Issue**: `np.ones_like()` is used but `numpy` is not imported
- **Fix**: Added `import numpy as np`

---

### 5. **Invalid Indexing of z_to_populations Result**

- **File**: [echospec/simulation/spectroscopy_vs_amplitude.py](echospec/simulation/spectroscopy_vs_amplitude.py)
- **Line**: 121
- **Issue**: Code assumes `z_to_populations()` returns a tuple/list with at least 2 elements, but type hints indicate it may not
- **Error**: `"__getitem__" method not defined on type "Tuple"`
- **Fix**: Added type checking before indexing: `pop_result[1] if isinstance(pop_result, (tuple, list)) else pop_result`

---

### 6. **Return Type Mismatch in run() Method**

- **File**: [echospec/simulation/spectroscopy_vs_amplitude.py](echospec/simulation/spectroscopy_vs_amplitude.py)
- **Line**: 83
- **Issue**: Method is annotated to return `xr.DataArray` but can return `None`
- **Error**: `Type "Unknown | DataArray | None" is not assignable to return type "DataArray"`
- **Fixes**:
  1. Updated docstring to indicate `xr.DataArray | None`
  2. Added check for empty `spectra` list before concatenation
  3. Updated call to `plot_final_z()` to check for `None`

---

## Files Modified

1. ✅ [echospec/simulation/spectroscopy.py](echospec/simulation/spectroscopy.py)
2. ✅ [echospec/simulation/hamiltonian.py](echospec/simulation/hamiltonian.py)
3. ✅ [echospec/simulation/spectroscopy_vs_amplitude.py](echospec/simulation/spectroscopy_vs_amplitude.py)

## Verification

All files now:

- ✅ Compile without syntax errors
- ✅ Pass type checking for missing imports
- ✅ Handle None cases safely
- ✅ Have corrected annotations
