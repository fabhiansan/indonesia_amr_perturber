1.
Kesalahan Tingkat Bingkai Semantik (Semantic Frame Errors): Kesalahan yang berkaitan dengan representasi skematis suatu peristiwa, relasi, atau keadaan, yang terdiri dari predikat dan partisipan (elemen bingkai). Tingkat ini mencakup kesalahan pada bingkai itu sendiri, elemen bingkai inti, dan elemen bingkai non-inti.
◦
Predicate Error (PredE): Kesalahan di mana predikat (kata kerja atau frasa yang menggambarkan tindakan atau keadaan) dalam pernyataan ringkasan tidak konsisten dengan teks sumber. Secara lebih umum, ini mewakili kasus di mana bingkai dari pernyataan ringkasan tidak selaras dengan apa yang diungkapkan dalam teks sumber. Sumber lain menyebutnya sebagai "Relation Error".
▪
Contoh (dari Tabel 1):
•
Teks Sumber Asli (bagian relevan): "The first vaccine for Ebola was approved by the FDA in 2019..."
•
Pernyataan Ringkasan yang Mengandung PredE: "The Ebola vaccine was rejected by the FDA in 2019."
•
Penjelasan: Predikat "rejected" dalam ringkasan salah dan tidak sesuai dengan "approved" dalam sumber.
◦
Entity Error (EntE): Kesalahan di mana argumen utama (seperti entitas, subjek atau objek) dari predikat salah atau memiliki atribut yang salah, meskipun relasi (predikat) tersebut diekspresikan dalam teks asli. Ini mencakup kasus-kasus di mana elemen bingkai inti dalam bingkai salah, serta kesalahan arah (seperti pertukaran agen-pasien). Ini adalah salah satu jenis kesalahan yang paling sering muncul, baik di dataset CNN/DM maupun XSum. Sumber lain menyebutnya sebagai "Entity Error".
▪
Contoh (dari Tabel 1):
•
Teks Sumber Asli (bagian relevan): "The first vaccine for Ebola was approved by the FDA in 2019..."
•
Pernyataan Ringkasan yang Mengandung EntE: "The COVID-19 vaccine was approved by the FDA in 2019."
•
Penjelasan: Entitas "COVID-19" dalam ringkasan salah; entitas yang benar adalah "Ebola".
◦
Circumstance Error (CircE): Kesalahan di mana satu atau lebih atribut tambahan (elemen bingkai non-inti dalam bingkai), yang menjelaskan keadaan di sekitar argumen dan predikat (misalnya, lokasi, waktu, cara, arah, modalitas), salah. Sumber lain menyebutnya sebagai "Circumstance Error".
▪
Contoh (dari Tabel 1):
•
Teks Sumber Asli (bagian relevan): "The first vaccine for Ebola was approved by the FDA in 2019 in the US..."
•
Pernyataan Ringkasan yang Mengandung CircE: "The first vaccine for Ebola was approved by the FDA in 2014."
•
Penjelasan: Atribut waktu "2014" dalam ringkasan salah; waktu yang benar adalah "2019".
2.
Kesalahan Tingkat Wacana (Discourse Errors): Kesalahan yang muncul dari relasi antar bagian teks dan dapat melampaui satu bingkai semantik, seperti relasi kausalitas atau urutan temporal. Kategori-kategori di bawah tingkat ini tidak muncul di dataset XSum karena ringkasannya hanya satu kalimat.
◦
Coreference Error (CorefE): Kesalahan di mana pronomina dan jenis referensi lain ke entitas yang disebutkan sebelumnya salah atau tidak memiliki anteseden yang jelas, membuatnya ambigu. Ini adalah salah satu jenis kesalahan yang paling sering muncul di dataset CNN/DM. Anotator manusia memiliki kesulitan signifikan dalam mengidentifikasi kesalahan ini, sering bingung dengan "Not an Error".
▪
Contoh (dari Tabel 1):
•
Teks Sumber Asli (bagian relevan): "...The first vaccine for Ebola was approved in 2019... Scientists say a vaccine for COVID-19 is unlikely to be ready this year."
•
Pernyataan Ringkasan yang Mengandung CorefE: "The first vaccine for Ebola was approved in 2019. They say a vaccine for COVID-19 is unlikely to be ready this year."
•
Penjelasan: Pronomina "They" dalam ringkasan tidak memiliki anteseden yang jelas dalam ringkasan itu sendiri, meskipun "Scientists" muncul di sumber.
◦
Discourse Link Error (LinkE): Kesalahan yang melibatkan tautan wacana antar pernyataan yang berbeda. Ini termasuk kesalahan urutan temporal yang salah atau tautan wacana yang salah (misalnya, relasi RST, konektor wacana) antar pernyataan. Sumber lain menyebutnya sebagai "Connector". Kesalahan ini kadang bingung dengan PredE dan EntE oleh anotator.
▪
Contoh (dari Tabel 1):
•
Teks Sumber Asli (bagian relevan): "...To produce the vaccine, scientists had to sequence the DNA of Ebola, then identify possible vaccines, and finally show successful clinical trials."
•
Pernyataan Ringkasan yang Mengandung LinkE: "To produce the vaccine, scientists have to show successful human trials, then sequence the DNA of the virus."
•
Penjelasan: Urutan langkah-langkah dalam ringkasan salah ("show successful human trials" lalu "sequence the DNA") dibandingkan dengan urutan sebenarnya dalam sumber ("sequence the DNA" lalu "identify possible vaccines" lalu "show successful clinical trials"). Kata hubung "then" digunakan secara tidak tepat berdasarkan sumber.
3.
Kesalahan Keterverifikasian Konten (Content Verifiability Errors): Kesalahan di mana pernyataan dalam ringkasan tidak dapat diverifikasi terhadap teks sumber.
◦
Out of Article Error (OutE): Pernyataan berisi informasi yang tidak ada dalam artikel sumber. Ringkasan seharusnya hanya berisi informasi yang dapat disimpulkan dari teks asli. Pekerjaan sebelumnya merujuk pada ini sebagai halusinasi ekstrinsik (extrinsic hallucinations). Ini adalah salah satu jenis kesalahan yang paling sering muncul di dataset XSum. Anotator kadang bingung kesalahan ini dengan PredE dan CircE, menganggap relasi atau informasi tambahan tidak dapat diverifikasi dari sumber. Sumber lain menyebutnya sebagai "Not in article".
▪
Contoh (dari Tabel 1):
•
Teks Sumber Asli (tidak ada informasi ini)
•
Pernyataan Ringkasan yang Mengandung OutE: "China has already started clinical trials of the COVID-19 vaccine."
•
Penjelasan: Informasi tentang China memulai uji klinis vaksin COVID-19 tidak ada dalam teks sumber yang diberikan.
◦
Grammatical Error (GramE): Tata bahasa kalimat sangat salah sehingga menjadi tidak bermakna atau tidak dapat dipahami. Ketika kesalahan tata bahasa membuat makna pernyataan tidak dapat dipahami atau ambigu, maka dianggap salah secara faktual karena tidak dapat diverifikasi terhadap sumber; kesalahan tata bahasa minor dapat diterima. Model berbasis LSTM cenderung memiliki proporsi GramE yang lebih tinggi. Sumber lain menyebutnya sebagai "Grammar".
▪
Contoh (dari Tabel 1):
•
Teks Sumber Asli (bagian relevan): "The Ebola vaccine was approved by the FDA in 2019..."
•
Pernyataan Ringkasan yang Mengandung GramE: "The Ebola vaccine accepted have already started."
•
Penjelasan: Struktur tata bahasa kalimat ini salah dan membuatnya sulit atau tidak mungkin dipahami maknanya, sehingga tidak dapat diverifikasi terhadap sumber.