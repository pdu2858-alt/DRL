## 1. Project Restructuring

- [ ] 1.1 Create `src/`, `models/`, and `data/` directories.
- [ ] 1.2 Delete the unused `final.py` script.
- [ ] 1.3 Move all remaining Python scripts (`download_dataset.py`, `train_env_model.py`, `patient_gym_env.py`, `train_agent.py`, `evaluate_agent.py`) into the `src/` directory.

## 2. Configuration Management

- [ ] 2.1 Create `src/config.py` to hold constants (e.g., model paths, hyperparameters, BIS bounds).
- [ ] 2.2 Update `src/download_dataset.py` to use config values.
- [ ] 2.3 Update `src/train_env_model.py` to use config values.
- [ ] 2.4 Update `src/patient_gym_env.py` to use config values.
- [ ] 2.5 Update `src/train_agent.py` and `src/evaluate_agent.py` to use config values.

## 3. Logging & Type Hinting

- [ ] 3.1 Import and configure the standard `logging` module in all scripts within `src/` to replace `print()` statements.
- [ ] 3.2 Add static type hints (`typing`) to functions and classes in `src/train_env_model.py` and `src/patient_gym_env.py`.

## 4. Finalization

- [ ] 4.1 Create `requirements.txt` in the root directory listing all dependencies.
- [ ] 4.2 Verify paths and imports work correctly by doing a dry-run or syntax check of the scripts.
