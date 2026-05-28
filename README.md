# Statistical Audit of pandas-dev/pandas

## Project Description
Proyek ini melakukan audit statistik terhadap repositori GitHub open-source `pandas-dev/pandas`.
Analisis ini menerapkan estimasi statistik, interval kepercayaan, pengujian hipotesis, dan teknik simulasi komputasi untuk mengevaluasi kesehatan repositori, aktivitas kontributor, dan kinerja manajemen isu.

Proyek ini menggunakan data peristiwa GitHub nyata seperti isu, permintaan tarik (pull request), stempel waktu, dan aktivitas penggabungan (merge) yang dikumpulkan melalui GitHub REST API.

Tugas ini diselesaikan sebagai bagian dari proyek kelompok akhir untuk mata kuliah Statistika & Probabilitas Sistem dan Teknologi Informasi.
---

## Repository Analyzed
Repository: pandas-dev/pandas
Repository URL: https://github.com/pandas-dev/pandas

---

## Team Members
| Member | Name | NIM | Role |
|---|---|
| A | Ahmad Aqil Fadria | 1519625006 |  Data Engineer |
| B | Nasya Putri Salsabila | 1519625007 | Estimation Analyst |
| C | Muhammad Hanief Inayatur Rahman | ... | Inference Analyst |
| D | Muhammad Rizqi Hazami | ... | Hypothesis Analyst |
| E | Krishna Dhikha Pratama | ... | Computation Analyst |

---

## Research Questions
1. Berapa estimasi probabilitas sebuah pull request dapat di-merge pada repository pandas-dev/pandas?
2. Apakah rata-rata tingkat penyelesaian issue pada repository pandas-dev/pandas berubah secara signifikan dalam periode tertentu?
3. Berapa probabilitas sebuah issue membutuhkan waktu lebih dari 30 hari untuk ditutup berdasarkan pendekatan simulasi statistik?

---

## Temuan Utama

Akan diisi setelah seluruh analisis selesai.

---

## Struktur Reposity
stat-audit-moby-sti-2025/ → README.md → AI_USAGE_LOG.md → data/ → raw/ (data asli GitHub API) → clean/ (dataset.csv, pull_requests.csv) → src/ → estimator.py [Member B] → inference.py [Member C] → hypothesis.py [Member D] → simulation.py [Member E] → notebooks/ → 01_eda.ipynb → 02_estimation.ipynb → 03_confidence_interval.ipynb → 04_hypothesis_testing.ipynb → 05_simulation.ipynb → report/ → statistical_health_report.pdf → presentation/ → video_link.md → requirements.txt
