# Deviations from the spec

- Project built directly in workspace root (/Users/jenilbuha/Desktop/sem7/Mlops/assign) per user confirmation.
- pyarrow pinned to 15.0.2 instead of 16.1.0 because mlflow 2.14.1 requires pyarrow<16,>=4.0.0.
- pathspec pinned to 0.12.1 because pathspec 1.x removed _DIR_MARK breaking dvc 3.51.2.
