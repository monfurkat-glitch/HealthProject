# Dataset

**Medical Appointment No Shows**, by Joni Hoppen and Aquarela Analytics, published on Kaggle:
<https://www.kaggle.com/datasets/joniarroba/noshowappointments>

| Property | Value |
|---|---|
| File | `KaggleV2-May-2016.csv` |
| Rows | 110,527 appointments (one row per appointment) |
| Columns | 14 |
| Patients | 62,299 unique `PatientId` values |
| Period | Scheduled 2015-11-10 to 2016-06-08; appointments 2016-04-29 to 2016-06-08 |
| Location | Public health clinics in Vitória, Espírito Santo, Brazil |
| SHA-256 | `9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d` |

## License and attribution

The dataset is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
It is redistributed here unmodified, for non-commercial educational use, so that the project can be reproduced in a clean Google Colab runtime.
All credit for collecting and publishing the data goes to the original authors.

## Column dictionary

| Column | Type | Description |
|---|---|---|
| `PatientId` | float (ID) | De-identified patient identifier. Read as a string; do not use as a numeric feature |
| `AppointmentID` | int (ID) | Unique appointment identifier |
| `Gender` | `F` / `M` | Patient gender |
| `ScheduledDay` | datetime (UTC) | When the appointment was booked (date and time) |
| `AppointmentDay` | date | Day of the appointment (no time of day recorded) |
| `Age` | int | Patient age in years |
| `Neighbourhood` | string | Neighbourhood of the clinic (81 values) |
| `Scholarship` | 0 / 1 | Enrolled in the *Bolsa Família* welfare programme |
| `Hipertension` | 0 / 1 | Has hypertension (original spelling) |
| `Diabetes` | 0 / 1 | Has diabetes |
| `Alcoholism` | 0 / 1 | Has alcoholism |
| `Handcap` | 0–4 | Number of disabilities (original spelling) |
| `SMS_received` | 0 / 1 | At least one SMS reminder was sent |
| `No-show` | `Yes` / `No` | **Target.** `Yes` means the patient did **not** attend |

## Known data issues

| Issue | Count |
|---|---|
| `Age = -1` | 1 row |
| `ScheduledDay` after `AppointmentDay` | 5 rows |
| `Handcap` values above 1 | 199 rows |
| Neighbourhoods with fewer than 50 appointments | 5 (smallest: `PARQUE INDUSTRIAL`, 1 row) |
| Class imbalance | 20.2% no-show |
| Appointments on a Saturday | 39 rows |
