"""
Jalankan dengan:
    python processing.py data_mentah.csv data_bersih.csv
    python processing.py data_mentah.csv data_bersih.csv --log log_perubahan.csv
"""
import argparse
import csv
import io
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HEADER_DIHARAPKAN = 12                 # jumlah kolom pada data mentah

KECAMATAN_VALID = {
    "Andir", "Antapani", "Arcamanik", "Astanaanyar", "Babakan Ciparay", "Bandung Kidul",
    "Bandung Kulon", "Bandung Wetan", "Batununggal", "Bojongloa Kaler", "Bojongloa Kidul",
    "Buahbatu", "Cibeunying Kaler", "Cibeunying Kidul", "Cibiru", "Cicendo", "Cidadap",
    "Cinambo", "Coblong", "Gedebage", "Kiaracondong", "Lengkong", "Mandalajati",
    "Panyileukan", "Rancasari", "Regol", "Sukajadi", "Sukasari", "Sumur Bandung", "Ujungberung",
}

EJAAN_KELURAHAN = {"Pasir Biru": "Pasirbiru", "Pasir Jati": "Pasirjati"}

PERUSAHAAN = {
    "PT. Telkomsel": "Telkomsel",
    "PT. Protelindo, Tbk": "Protelindo",
    "PT. Protelindo": "Protelindo",
    "PT. Xl Axiata, Tbk": "XL Axiata",
    "PT. Solusi Tunas Pratama": "Solusi Tunas Pratama",
    "PT. Tower Bersama Group": "Tower Bersama Group",
    "Daya Mitra Telekomunikasi": "Daya Mitra Telekomunikasi",
    "PT. Indosat, Tbk": "Indosat",
    "PT. Centratama Menara Indonesia": "Centratama Menara Indonesia",
    "Naragita": "Naragita",
    "Hutchinson": "Hutchison",
    "PT. Permata Karya Perdana": "Permata Karya Perdana",
    "Telkom": "Telkom",                 
}

STRUKTUR = {                           
    "pole": "Pole", "monopole": "Monopole",
    "4 kaki": "Tower 4 Kaki", "kaki 4": "Tower 4 Kaki", "4": "Tower 4 Kaki",
    "3 kaki": "Tower 3 Kaki", "kaki 3": "Tower 3 Kaki",
    "sst": "SST", "mcp": "Microcell Pole", "microcell pole": "Microcell Pole",
    "camouflage": "Kamuflase", "kamuflase pohon": "Kamuflase",
    "rooftop mt": "Rooftop", "pole rooftop": "Pole",
}
STRUKTUR_ROOFTOP = {"rooftop mt", "pole rooftop"}
STRUKTUR_STANDAR = set(STRUKTUR.values())

STATUS_IMB = {"Ada nomor IMB", "Sudah terbit", "Belum terbit", "Dalam proses",
              "Tidak diperlukan", "Tidak ada data"}

KOLOM_AKHIR = ["id", "kecamatan", "kelurahan", "nama_perusahaan", "lokasi_menara", "latitude",
               "longitude", "tinggi_m", "struktur_tower", "penempatan", "status_imb",
               "nomor_imb", "keterangan", "catatan_kualitas"]
KOLOM_TEKS = ["kecamatan", "kelurahan", "nama_perusahaan", "lokasi_menara", "struktur_tower",
              "penempatan", "status_imb", "nomor_imb", "keterangan"]

FLAG_DIKENAL = ("KECAMATAN_TIDAK_VALID", "KECAMATAN_PERLU_VERIFIKASI", "KOORDINAT_RUSAK_PERLU_VERIFIKASI",
                "TINGGI_DIJUMLAHKAN_STACKING", "TINGGI_TIDAK_VALID", "LOKASI_KOSONG",
                "DUPLIKAT_KEMUNGKINAN", "PERUSAHAAN_BELUM_DIPETAKAN",
                "STRUKTUR_BELUM_DIPETAKAN", "BARIS_DIBACA_DARI_KANAN")

RINCIAN = False


class Pencatat:

    def __init__(self):
        self.log = []
        self.flag = defaultdict(list)

    def ubah(self, i, kolom, lama, baru, alasan):
        if str(lama) != str(baru):
            self.log.append(dict(id=i, kolom=kolom, nilai_lama=lama, nilai_baru=baru, alasan=alasan))

    def tandai(self, i, teks):
        self.flag[i].append(teks)


def judul(no, teks, penjelasan):
    print(f"\n[{no}/10] {teks}")
    if RINCIAN:
        print(f"       {penjelasan}")


def info(teks):
    if RINCIAN or teks.startswith("Perlu dicek:"):
        print(f"       - {teks}")


def bersihkan_teks(s):
    if s is None:
        return ""
    s = s.replace("\xa0", " ").replace("Ê", " ").replace("Ð", "-")
    return re.sub(r"\s+", " ", s).strip()


def angka(s):
    return bool(re.fullmatch(r"\s*-?\d+([.,]\d+)?\s*", s or ""))


def ke_float(s):
    try:
        return float(str(s).replace(",", "."))
    except ValueError:
        return np.nan


def ok_lat(v):
    return np.isfinite(v) and -90 <= v <= 90


def ok_lon(v):
    return np.isfinite(v) and -180 <= v <= 180


def gabung_koma_desimal(r):
    out, j, n = list(r[:5]), 5, 0
    while j < len(r):
        if j + 1 < len(r) and re.fullmatch(r"-\d{1,3}", r[j].strip()) and re.fullmatch(r"\d{3,}", r[j + 1].strip()):
            out.append(f"{r[j].strip()},{r[j + 1].strip()}")
            j, n = j + 2, n + 1
        else:
            out.append(r[j])
            j += 1
    return out, n


def langkah1_baca(path, p):
    judul(1, "Membaca CSV mentah",
          "Koma di dalam nilai bisa membuat kolom bergeser; baris seperti itu disusun kembali.")
    try:
        isi = Path(path).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        isi = Path(path).read_text(encoding="cp1252")
        info("File bukan UTF-8; dibaca dengan encoding cp1252.")
    semua = [r for r in csv.reader(io.StringIO(isi)) if any(x.strip() for x in r)]
    header, baris = semua[0], semua[1:]
    info(f"Terbaca {len(baris)} baris dengan {len(header)} kolom pada header.")
    if len(header) != HEADER_DIHARAPKAN:
        info(f"Perlu dicek: header seharusnya {HEADER_DIHARAPKAN} kolom, tetapi ada {len(header)}.")

    salah = sum(1 for r in baris if len(r) != len(header))
    info(f"Ada {salah} baris dengan jumlah kolom berbeda dari header.")

    data, n_koma, n_kanan = [], 0, 0
    for r in baris:
        r, n = gabung_koma_desimal(r)
        n_koma += 1 if n else 0
        idx = None
        for i in range(5, len(r) - 6):
            if all(angka(x) for x in r[i:i + 4]) and abs(ke_float(r[i])) > 1 and abs(ke_float(r[i + 1])) > 100:
                idx = i
                break
        dari_kanan = False
        if idx is None:                       
            idx, dari_kanan = len(r) - 7, True
            n_kanan += 1
            if idx < 5:
                raise ValueError(f"Baris tidak bisa dibaca (id {r[0]}): {r}")
        perusahaan, mulai = r[3], 4
        if r[4].strip().lower() == "tbk":      
            perusahaan, mulai = f"{r[3].strip()}, Tbk", 5
        data.append(dict(
            id=int(r[0]), kecamatan=r[1], kelurahan=r[2], perusahaan=perusahaan,
            lokasi=",".join(r[mulai:idx]),
            lat_rev=r[idx], lon_rev=r[idx + 1], lon_raw=r[idx + 2], lat_raw=r[idx + 3],
            tinggi=r[idx + 4], struktur=r[idx + 5], imb=",".join(r[idx + 6:]),
            dari_kanan=dari_kanan))
    df = pd.DataFrame(data)
    info(f"Nama perusahaan, lokasi, dan IMB yang terpisah koma disatukan pada {salah} baris.")
    info(f"Koma desimal pada latitude diperbaiki di {n_koma} baris.")
    if n_kanan:
        info(f"Perlu dicek: {n_kanan} baris dibaca dari kanan karena pola koordinatnya tidak ditemukan.")
        for i in df[df.dari_kanan].id:
            p.tandai(i, "BARIS_DIBACA_DARI_KANAN")
    tertukar = ((df.lat_rev == df.lon_raw) & (df.lon_rev == df.lat_raw) & (df.lat_rev != df.lat_raw)).sum()
    info(f"Kolom Latitude/Longitude asli tertukar pada {tertukar} baris. Nilai dari kolom *_rev dipakai.")
    return df


def langkah2_teks(df, p):
    judul(2, "Merapikan teks", "Spasi berlebih dan karakter rusak dibersihkan dari kolom teks.")
    total = 0
    for col in ["kecamatan", "kelurahan", "perusahaan", "lokasi", "tinggi", "struktur", "imb"]:
        baru = df[col].map(bersihkan_teks)
        beda = int((baru != df[col]).sum())
        total += beda
        if beda:
            info(f"{col}: {beda} nilai dirapikan.")
        df[col] = baru
    info(f"Total {total} sel dirapikan.")


def langkah3_wilayah(df, p):
    judul(3, "Menyeragamkan kecamatan & kelurahan",
          "Ejaan kelurahan disamakan dan nama kecamatan yang tidak valid diperiksa.")
    n_ejaan = 0
    for k, row in df.iterrows():
        baru = EJAAN_KELURAHAN.get(row.kelurahan, row.kelurahan)
        if baru != row.kelurahan:
            p.ubah(row.id, "kelurahan", row.kelurahan, baru, "Ejaan kelurahan disamakan tanpa spasi.")
            df.at[k, "kelurahan"] = baru
            n_ejaan += 1
    info(f"Ejaan kelurahan diperbaiki pada {n_ejaan} baris.")

    valid = df[df.kecamatan.isin(KECAMATAN_VALID)]
    n_fix = n_flag = 0
    for k, row in df[~df.kecamatan.isin(KECAMATAN_VALID)].iterrows():
        kandidat = valid[valid.kelurahan == row.kelurahan].kecamatan.value_counts()
        if len(kandidat) and kandidat.iloc[0] / kandidat.sum() >= 0.75:
            baru = kandidat.index[0]
            p.ubah(row.id, "kecamatan", row.kecamatan, baru,
                   f"Kecamatan awal tidak valid. Dari {kandidat.sum()} baris di {row.kelurahan}, {kandidat.iloc[0]} berada di {baru}.")
            df.at[k, "kecamatan"] = baru
            n_fix += 1
        else:
            p.tandai(row.id, "KECAMATAN_TIDAK_VALID")
            n_flag += 1
    info(f"Kecamatan diperbaiki pada {n_fix} baris berdasarkan kelurahan yang sama.")
    if n_flag:
        info(f"Ada {n_flag} kecamatan yang belum bisa dipastikan; barisnya ditandai.")

    n_verif = 0
    for kel, g in df.groupby("kelurahan"):
        vc = g.kecamatan.value_counts()
        if len(g) >= 4 and len(vc) > 1 and vc.iloc[0] / len(g) >= 0.75:
            for i in g[g.kecamatan != vc.index[0]].id:
                p.tandai(i, f"KECAMATAN_PERLU_VERIFIKASI(mayoritas {kel}={vc.index[0]})")
                n_verif += 1
    info(f"Ada {n_verif} baris dengan kecamatan berbeda dari mayoritas di kelurahannya. Baris ini ditandai untuk dicek.")


def langkah4_perusahaan(df, p):
    judul(4, "Menyeragamkan nama perusahaan",
          "Awalan PT., akhiran Tbk, dan variasi ejaan perusahaan dirapikan.")
    n_ubah, n_baru = 0, 0
    for k, row in df.iterrows():
        baru = PERUSAHAAN.get(row.perusahaan)
        if baru is None:                     
            baru = re.sub(r"^PT\.?\s*", "", row.perusahaan, flags=re.I)
            baru = re.sub(r",?\s*Tbk\.?$", "", baru, flags=re.I).strip()
            p.tandai(row.id, "PERUSAHAAN_BELUM_DIPETAKAN")
            n_baru += 1
        if baru != row.perusahaan:
            p.ubah(row.id, "nama_perusahaan", row.perusahaan, baru,
                   "Awalan PT., akhiran Tbk, atau ejaan nama perusahaan dirapikan.")
            n_ubah += 1
        df.at[k, "perusahaan"] = baru
    info(f"Sebanyak {n_ubah} nama dirapikan. Ada {df.perusahaan.nunique()} nama perusahaan setelahnya.")
    if n_baru:
        info(f"Perlu dicek: {n_baru} nama belum ada di daftar perusahaan. Namanya dirapikan dan barisnya ditandai.")


def langkah5_koordinat(df, p):
    judul(5, "Memperbaiki koordinat",
          "Koordinat diambil dari kolom *_rev. Salah tulis yang jelas diperbaiki.")
    st = defaultdict(int)
    lat_out, lon_out = [], []
    for k, row in df.iterrows():
        lat, lon = ke_float(row.lat_rev), ke_float(row.lon_rev)
        lat0, lon0 = row.lat_rev, row.lon_rev

        if abs(lat) > 90:
            digit = re.sub(r"\D", "", lat0)
            kor = float(digit[0] + "." + digit[1:]) * (-1 if lat < 0 else 1)
            if ok_lat(kor):
                p.ubah(row.id, "latitude", lat0, kor,
                       "Pemisah desimal latitude keliru; titik dipindah setelah angka pertama.")
                lat = kor
                st["lat_desimal"] += 1
        if lon > 1000:                      
            kor = lon / 10 ** (len(str(int(abs(lon)))) - 3)
            if ok_lon(kor):
                p.ubah(row.id, "longitude", lon0, kor, "Titik desimal longitude ditambahkan setelah tiga angka pertama.")
                lon = kor
                st["lon_desimal"] += 1
        if not (ok_lat(lat) and ok_lon(lon)):  
            p.ubah(row.id, "latitude/longitude",
                   f"lat={lat0}; lon={lon0} (kolom lain: lat={row.lat_raw}, lon={row.lon_raw})",
                   "(kosong)", "Koordinat tidak bisa dipastikan dari data yang tersedia.")
            p.tandai(row.id, "KOORDINAT_RUSAK_PERLU_VERIFIKASI")
            lat = lon = np.nan
            st["kosong"] += 1
        lat_out.append(round(lat, 6) if not np.isnan(lat) else np.nan)
        lon_out.append(round(lon, 6) if not np.isnan(lon) else np.nan)
    df["latitude"], df["longitude"] = lat_out, lon_out
    info(f"Pemisah desimal latitude diperbaiki pada {st['lat_desimal']} baris.")
    info(f"Titik desimal longitude ditambahkan pada {st['lon_desimal']} baris.")
    info(f"Koordinat pada {st['kosong']} baris dikosongkan karena belum bisa dipastikan.")
    info("Koordinat disimpan hingga enam angka di belakang koma.")


def langkah6_struktur(df, p):
    judul(6, "Menyeragamkan struktur tower", "Variasi nama struktur disatukan dan informasi rooftop dicatat.")
    rooftop, n_ubah, n_baru = [], 0, 0
    for k, row in df.iterrows():
        kunci = row.struktur.lower()
        baru = STRUKTUR.get(kunci)
        if baru is None:
            baru = row.struktur.strip().title()
            p.tandai(row.id, "STRUKTUR_BELUM_DIPETAKAN")
            n_baru += 1
        if baru != row.struktur:
            p.ubah(row.id, "struktur_tower", row.struktur, baru, "Penulisan jenis struktur disamakan.")
            n_ubah += 1
        df.at[k, "struktur"] = baru
        rooftop.append(kunci in STRUKTUR_ROOFTOP)
    df["rooftop"] = rooftop
    info(f"Sebanyak {n_ubah} nilai dirapikan. Ada {df.struktur.nunique()} jenis struktur setelahnya.")
    if n_baru:
        info(f"Perlu dicek: {n_baru} baris memiliki jenis struktur yang belum ada di daftar.")


def langkah7_tinggi(df, p):
    judul(7, "Mengubah tinggi menjadi angka", "Nilai tinggi diubah menjadi angka dalam meter.")
    out, st = [], defaultdict(int)
    for k, row in df.iterrows():
        s, val = row.tinggi, np.nan
        if re.fullmatch(r"\d+(\.\d+)?", s):
            val = float(s)
        elif (m := re.fullmatch(r"RT\s*(\d+(?:\.\d+)?)\s*m", s, re.I)):
            val = float(m.group(1))
            df.at[k, "rooftop"] = True
            p.ubah(row.id, "tinggi_m", s, f"{val:g}", "Angka tinggi diambil; awalan RT dicatat sebagai penempatan rooftop.")
            st["rt"] += 1
        elif (m := re.fullmatch(r"(\d+)\s*\+\s*(\d+)\s*\(stacking\)", s, re.I)):
            val = float(m.group(1)) + float(m.group(2))
            p.ubah(row.id, "tinggi_m", s, f"{val:g}", "Tinggi dasar dan tambahan stacking dijumlahkan.")
            p.tandai(row.id, "TINGGI_DIJUMLAHKAN_STACKING")
            st["stacking"] += 1
        else:
            p.ubah(row.id, "tinggi_m", s, "(kosong)", "Nilai tinggi tidak bisa dibaca sebagai angka.")
            p.tandai(row.id, "TINGGI_TIDAK_VALID")
            st["invalid"] += 1
        out.append(val)
    df["tinggi_m"] = out
    info(f"Awalan RT ditemukan pada {st['rt']} baris; angkanya disimpan sebagai tinggi dan penempatannya menjadi Rooftop.")
    info(f"Tambahan stacking dijumlahkan pada {st['stacking']} baris.")
    info(f"Ada {st['invalid']} nilai tinggi yang tidak jelas; nilainya dikosongkan dan barisnya ditandai.")


def langkah8_imb(df, p):
    judul(8, "Memisahkan data IMB", "Isi kolom IMB dibagi menjadi status, nomor, dan keterangan.")
    status, nomor, ket = [], [], []
    for k, row in df.iterrows():
        s, st, no, kt = row.imb, "", "", ""
        if s in ("", "-", "0"):
            st = "Tidak ada data"
        elif re.fullmatch(r"NO NEED|Not Required|IMB NR|tidak diperlukan IMB", s, re.I):
            st = "Tidak diperlukan"
        elif re.fullmatch(r"Proses", s, re.I):
            st = "Dalam proses"
        elif re.fullmatch(r"IMB RELEASED", s, re.I):
            st = "Sudah terbit"
        elif re.fullmatch(r"IMB NY RELEASED", s, re.I):
            st = "Belum terbit"
        elif s.upper().startswith("REKOM DISHUBKOMINFO"):
            st, kt = "Dalam proses", "Rekomendasi Dishubkominfo & KRK selesai, menunggu pengesahan IMB"
        elif re.fullmatch(r"ROOFTOP \d+ ?m", s, re.I):
            st = "Tidak ada data"
            df.at[k, "rooftop"] = True
        elif re.search(r"\d{2,}", s):
            st, no = "Ada nomor IMB", re.sub(r"^IMB\s*-\s*", "", s)
        else:                                
            st, kt = "Tidak ada data", s
        if not (st == "Ada nomor IMB" and no == s):
            p.ubah(row.id, "data_imb", row.imb, f"status={st}; nomor={no or '-'}", "Status dan nomor IMB dipisahkan.")
        status.append(st)
        nomor.append(no)
        ket.append(kt)
    df["status_imb"], df["nomor_imb"], df["keterangan"] = status, nomor, ket
    for nama, n in df.status_imb.value_counts().items():
        info(f"Status '{nama}': {n} baris.")

def langkah9_flag(df, p):
    judul(9, "Menandai data yang perlu dicek", "Lokasi kosong dan kemungkinan data ganda diberi catatan.")
    n_lok = 0
    for k, row in df.iterrows():
        if row.lokasi in ("", "-"):
            p.ubah(row.id, "lokasi_menara", row.lokasi, "(kosong)", "Lokasi hanya berisi tanda minus.")
            df.at[k, "lokasi"] = ""
            p.tandai(row.id, "LOKASI_KOSONG")
            n_lok += 1
    info(f"Ada {n_lok} lokasi yang hanya berisi tanda minus; nilainya dikosongkan.")

    pertama, n_dup = {}, 0
    for _, row in df.dropna(subset=["latitude", "longitude"]).iterrows():
        kunci = (row.perusahaan, row.latitude, row.longitude)
        if kunci in pertama:
            p.tandai(row.id, f"DUPLIKAT_KEMUNGKINAN(sama dengan id {pertama[kunci]})")
            n_dup += 1
        else:
            pertama[kunci] = row.id
    info(f"Ada {n_dup} kemungkinan data ganda karena perusahaan dan koordinatnya sama; barisnya ditandai.")
    info("Koordinat yang dipakai perusahaan berbeda tetap dianggap sebagai data terpisah.")

    df["penempatan"] = np.where(df.rooftop, "Rooftop", "Tidak disebutkan")
    df["catatan_kualitas"] = df.id.map(lambda i: "; ".join(p.flag[i]))


def langkah10_simpan(df, p, out, log_path):
    judul(10, "Menyimpan hasil", "Kolom disusun, baris diurutkan berdasarkan ID, lalu file CSV ditulis.")
    final = df.rename(columns={"perusahaan": "nama_perusahaan", "lokasi": "lokasi_menara",
                               "struktur": "struktur_tower"})[KOLOM_AKHIR].sort_values("id").reset_index(drop=True)
    final["latitude"] = final.latitude.map(lambda v: "" if pd.isna(v) else repr(float(v)))
    final["longitude"] = final.longitude.map(lambda v: "" if pd.isna(v) else repr(float(v)))
    final["tinggi_m"] = final.tinggi_m.map(lambda v: "" if pd.isna(v) else f"{v:g}")
    final["id"] = final.id.astype(str)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out, index=False, encoding="utf-8-sig", lineterminator="\n")
    info(f"Data bersih disimpan di {out} ({len(final)} baris, {len(final.columns)} kolom).")
    if log_path:
        pd.DataFrame(p.log, columns=["id", "kolom", "nilai_lama", "nilai_baru", "alasan"]) \
            .sort_values(["id", "kolom"]).to_csv(log_path, index=False, encoding="utf-8-sig", lineterminator="\n")
        info(f"Catatan perubahan disimpan di {log_path} ({len(p.log)} entri).")
    return final


def validasi(df_mentah, out, p):
    print("\nMemeriksa hasil...")
    hasil = []

    def cek(nama, status, detail=""):
        hasil.append((nama, status))
        tanda = {"LULUS": "[LULUS]     ", "PERINGATAN": "[PERINGATAN]", "GAGAL": "[GAGAL]     "}[status]
        if RINCIAN or status != "LULUS":
            print(f"{tanda} {nama}" + (f": {detail}" if detail else ""))

    mentah_id = set(df_mentah.id)
    f = pd.read_csv(out, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    rows = list(csv.reader(open(out, encoding="utf-8-sig", newline="")))

    cek("1. Jumlah baris sama dengan data mentah",
        "LULUS" if len(f) == len(df_mentah) else "GAGAL", f"{len(df_mentah)} menjadi {len(f)}")
    ids = f.id.astype(int)
    ok = ids.is_unique and ids.is_monotonic_increasing and set(ids) == mentah_id
    cek("2. ID unik, berurutan, dan sama dengan data mentah", "LULUS" if ok else "GAGAL")
    ukuran = {len(r) for r in rows}
    cek("3. Jumlah kolom konsisten pada setiap baris",
        "LULUS" if ukuran == {len(KOLOM_AKHIR)} else "GAGAL", f"jumlah kolom: {sorted(ukuran)}")
    buruk = sum(f[c].str.contains("[\xa0ÊÐ\ufffd]", regex=True).sum() for c in f.columns)
    cek("4. Tidak ada karakter rusak dalam teks", "LULUS" if buruk == 0 else "GAGAL", f"{buruk} sel")
    spasi = sum(((f[c] != f[c].str.strip()) | f[c].str.contains(r"\s{2,}", regex=True)).sum() for c in KOLOM_TEKS)
    cek("5. Tidak ada spasi berlebih di kolom teks", "LULUS" if spasi == 0 else "GAGAL", f"{spasi} sel")
    lat, lon = pd.to_numeric(f.latitude, errors="coerce"), pd.to_numeric(f.longitude, errors="coerce")
    sepasang = ((f.latitude == "") == (f.longitude == "")).all()
    valid = ((f.latitude.eq("") | lat.map(ok_lat)).all() and
             (f.longitude.eq("") | lon.map(ok_lon)).all())
    cek("6. Koordinat valid dan latitude/longitude terisi berpasangan",
        "LULUS" if (sepasang and valid) else "GAGAL", f"{int(lat.isna().sum())} baris kosong")
    diubah = {e["id"] for e in p.log if e["kolom"] in ("latitude", "longitude", "latitude/longitude")}
    m = f.assign(idn=ids).merge(df_mentah[["id", "lat_rev", "lon_rev"]], left_on="idn", right_on="id", suffixes=("", "_m"))
    la, lo = m.lat_rev.map(ke_float).round(6), m.lon_rev.map(ke_float).round(6)
    salah = m[(m.latitude != "") & ~m.idn.isin(diubah) &
              ((pd.to_numeric(m.latitude) - la).abs().gt(1e-9) | (pd.to_numeric(m.longitude) - lo).abs().gt(1e-9))]
    cek("7. Perubahan koordinat dari data mentah tercatat",
        "LULUS" if len(salah) == 0 else "GAGAL", f"{len(diubah)} baris tercatat, {len(salah)} selisih belum tercatat")
    t = pd.to_numeric(f.tinggi_m.replace("", np.nan), errors="coerce")
    bukan_angka = int(((f.tinggi_m != "") & t.isna()).sum())
    cek("8. Tinggi terisi sebagai angka jika tersedia", "GAGAL" if bukan_angka else "LULUS",
        f"{bukan_angka} bukan angka, {int(t.isna().sum())} kosong")
    sisa = int(f.nama_perusahaan.str.contains(r"\bPT\b|Tbk", case=False, regex=True).sum())
    baru = int(f.catatan_kualitas.str.contains("PERUSAHAAN_BELUM_DIPETAKAN").sum())
    cek("9. Nama perusahaan sudah dirapikan dan dikenali",
        "GAGAL" if sisa else ("PERINGATAN" if baru else "LULUS"), f"{sisa} masih memakai PT/Tbk, {baru} belum ada di daftar")
    luar = f[~f.struktur_tower.isin(STRUKTUR_STANDAR)]
    cek("10. Jenis struktur sesuai daftar", "LULUS" if luar.empty else "PERINGATAN",
        f"{len(luar)} belum ada di daftar")
    s_ok = f.status_imb.isin(STATUS_IMB).all()
    n_ok = ((f.nomor_imb != "") == (f.status_imb == "Ada nomor IMB")).all()
    cek("11. Status IMB sesuai dengan isi nomor IMB",
        "LULUS" if (s_ok and n_ok) else "GAGAL")
    luar = f[~f.kecamatan.isin(KECAMATAN_VALID)]
    cek("12. Nama kecamatan ada dalam daftar 30 kecamatan Kota Bandung", "LULUS" if luar.empty else "PERINGATAN",
        f"{len(luar)} tidak valid")
    ok = f.penempatan.isin(["Rooftop", "Tidak disebutkan"]).all()
    cek("13. Nilai penempatan sesuai daftar", "LULUS" if ok else "GAGAL")
    semua = f.catatan_kualitas[f.catatan_kualitas != ""].str.split("; ").explode()
    asing = semua[~semua.str.startswith(FLAG_DIKENAL)]
    cek("14. Catatan kualitas memakai jenis penanda yang dikenal", "LULUS" if asing.empty else "GAGAL",
        f"{len(semua)} catatan pada {int((f.catatan_kualitas != '').sum())} baris")
    ok = list(f.columns) == KOLOM_AKHIR and all(re.fullmatch(r"[a-z][a-z0-9_]*", c) for c in f.columns)
    cek("15. Nama dan urutan kolom sesuai rancangan", "LULUS" if ok else "GAGAL")
    ok = f.shape == (len(df_mentah), len(KOLOM_AKHIR)) and f.id.map(str).eq(f.id).all()
    cek("16. File dapat dibaca ulang dengan jumlah baris dan kolom yang tepat", "LULUS" if ok else "GAGAL", str(f.shape))

    g = sum(1 for _, s in hasil if s == "GAGAL")
    w = sum(1 for _, s in hasil if s == "PERINGATAN")
    print(f"Hasil pemeriksaan: {len(hasil) - g - w} lulus, {w} peringatan, {g} gagal. " +
          ("Data siap dipakai." if g == 0 else "Periksa bagian yang gagal sebelum memakai data."))
    return g


def main():
    global RINCIAN
    ap = argparse.ArgumentParser(description="Bersihkan CSV data menara telekomunikasi Kota Bandung.")
    ap.add_argument("input", help="lokasi file CSV mentah")
    ap.add_argument("output", help="lokasi file CSV hasil pembersihan")
    ap.add_argument("--log", default=None, help="lokasi catatan perubahan (bawaan: log_perubahan.csv di folder output)")
    ap.add_argument("--verbose", action="store_true", help="tampilkan rincian setiap tahap dan pemeriksaan")
    a = ap.parse_args()
    RINCIAN = a.verbose
    log_path = a.log or str(Path(a.output).with_name("log_perubahan.csv"))

    print(f"Data mentah : {a.input}")
    print(f"Data bersih : {a.output}")

    p = Pencatat()
    df = langkah1_baca(a.input, p)
    n_input = len(df)
    langkah2_teks(df, p)
    langkah3_wilayah(df, p)
    langkah4_perusahaan(df, p)
    langkah5_koordinat(df, p)
    langkah6_struktur(df, p)
    langkah7_tinggi(df, p)
    langkah8_imb(df, p)
    langkah9_flag(df, p)
    langkah10_simpan(df, p, a.output, log_path)

    mentah = langkah_mentah_untuk_cek(a.input)
    gagal = validasi(mentah, a.output, p)
    sys.exit(1 if gagal else 0)


def langkah_mentah_untuk_cek(path):
    class Diam(Pencatat):
        pass
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return langkah1_baca(path, Diam())


if __name__ == "__main__":
    main()
