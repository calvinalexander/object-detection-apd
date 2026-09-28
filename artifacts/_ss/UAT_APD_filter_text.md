# Ruang Lingkup dan Ketentuan Pengujian

- Objek UAT adalah model AI pada satu use case: **Deteksi Pelanggaran Penggunaan APD
  beserta penyaringan event sebelum tampil di dashboard**.
- Pengujian bersifat black box: event dari perangkat terpasang di lokasi SPPG selama
  periode uji dibandingkan dengan hasil penilaian manual. Struktur kode internal tidak diuji.
- Termasuk UAT: status APD per orang, hasil penyaringan event, dan event yang tampil di
  dashboard.
- Tidak termasuk UAT: perangkat keras kamera, server, dan GPU, serta integrasi ke sistem
  produksi di luar alur penyaringan event.
- Lingkungan uji: filter berjalan di perangkat SPPG. Setiap event pelanggaran dari
  perangkat dinilai ulang oleh model sebelum dikirim ke dashboard.
- Penilaian manual dilakukan dan diverifikasi bersama Pemberi Kerja.
- Target akurasi minimum (tanda … %) mengikuti dokumen teknis yang telah disepakati.

# Data Uji

| Kode | Deskripsi Data Uji | Durasi / Jumlah | Ground Truth |
|---|---|---|---|
| V-APD-01 | Event pelanggaran APD dari perangkat SPPG selama periode uji, beserta status APD per orang hasil model | Tanggal … s.d. …, … event | Penilaian manual atas gambar orang pada setiap event |
| V-APD-02 | Gambar event yang diunduh dari perangkat, dikelompokkan menjadi ditandai dan diloloskan | … gambar (ditandai …, diloloskan …) | Penilaian manual pada V-APD-03 |
| V-APD-03 | Lembar kurasi manual atas gambar V-APD-02 | … baris | Diisi Pemberi Kerja: pelanggaran / bukan pelanggaran |
| V-APD-04 | Daftar event pelanggaran pada dashboard untuk periode yang sama | … event | Event yang ditandai model |

# Hasil Uji UAT

| NO | Use Case / Skenario Uji | Berhasil/Gagal | Diuji oleh **) | Tanggal Test |
|---|---|---|---|---|
| 1 | UC1.1 Klasifikasi Status APD per Orang | | | |
| 2 | UC1.2 Penyaringan Event Sebelum Tampil di Dashboard | | | |
| 3 | UC1.3 Kebenaran Event yang Diloloskan | | | |
| 4 | UC1.4 Kebenaran Event yang Ditandai | | | |

**1. Nama Uji : UC1.1 Klasifikasi Status APD per Orang**

Deskripsi Pengujian : Verifikasi model memberi status APD (tutup kepala, masker, sarung
tangan) untuk setiap event pelanggaran dari perangkat.

Kasus Uji :
- Data uji : V-APD-01
- Tampilkan status pada terminal perangkat, lalu cocokkan dengan gambar orang.
- Status yang diperiksa : PELANGGARAN / SESUAI / TIDAK DAPAT DINILAI
- Kondisi : orang berdiri, berjalan, menyamping, dan sebagian membelakangi kamera

Hasil yang diharapkan :
- Berhasil jika setiap event memiliki status dan status sesuai kondisi aktual ≥ … %.
  TIDAK DAPAT DINILAI tidak dihitung sebagai pelanggaran.
- Gagal jika ada event tanpa status atau kesesuaian di bawah batas.

Hasil aktual :

---

**2. Nama Uji : UC1.2 Penyaringan Event Sebelum Tampil di Dashboard**

Deskripsi Pengujian : Verifikasi hanya event yang ditandai model yang tampil di dashboard.

Kasus Uji :
- Data uji : V-APD-02 dan V-APD-04
- Cocokkan daftar event dashboard dengan event yang ditandai dan diloloskan model pada
  periode yang sama.

Hasil yang diharapkan :
- Berhasil jika semua event yang ditandai tampil di dashboard dan tidak ada event yang
  diloloskan ikut tampil.
- Gagal jika ada event diloloskan yang tampil, atau event ditandai yang tidak tampil.

Hasil aktual :

---

**3. Nama Uji : UC1.3 Kebenaran Event yang Diloloskan**

Deskripsi Pengujian : Verifikasi event yang ditahan model (tidak tampil di dashboard)
memang bukan pelanggaran.

Kasus Uji :
- Data uji : V-APD-02 (gambar diloloskan) dan V-APD-03
- Buka gambar yang diloloskan pada PC SPPG, lalu cocokkan kondisi orang dengan penilaian
  manual.
- Penilai : perwakilan Pemberi Kerja / petugas SPPG

Hasil yang diharapkan :
- Berhasil jika event yang benar bukan pelanggaran ≥ … % dari yang diperiksa.
- Gagal jika pelanggaran yang ikut tertahan melebihi batas.

Hasil aktual :

---

**4. Nama Uji : UC1.4 Kebenaran Event yang Ditandai**

Deskripsi Pengujian : Verifikasi event yang tampil di dashboard memang pelanggaran dan
jenis pelanggarannya sesuai.

Kasus Uji :
- Data uji : V-APD-04 dan V-APD-03
- Tampilkan event pelanggaran di dashboard pada PC SPPG, lalu cocokkan jenis pelanggaran
  dengan kondisi orang pada gambar dan penilaian manual.
- Penilai : perwakilan Pemberi Kerja / petugas SPPG

Hasil yang diharapkan :
- Berhasil jika event yang benar pelanggaran ≥ … % dan jenis pelanggarannya sesuai.
- Gagal jika orang ber-APD lengkap ikut ditandai melebihi batas, atau jenis pelanggaran
  tidak sesuai.

Hasil aktual :

---

*) Kolom Berhasil/Gagal, Diuji oleh, Tanggal Test, dan Hasil aktual diisi tulisan tangan saat pengujian.
**) Diuji oleh diisi nama perwakilan Pemberi Kerja atau petugas SPPG yang menyaksikan dan
memverifikasi hasil.

# Rekapitulasi Hasil Uji

| Use Case | Jumlah Skenario | Berhasil | Gagal | Persentase Berhasil |
|---|---|---|---|---|
| Deteksi Pelanggaran APD dan Penyaringan Event | 4 | | | … % |
| Total | 4 | | | … % |

Catatan :
