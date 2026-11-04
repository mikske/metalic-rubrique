require "json"
require_relative '../lib/rubrique/config'
require_relative '../lib/rubrique/dataset'
include Rubrique

vocab = JSON.parse(File.read(Config::VOCAB_PATH))

%w[train val test].each do |split|
  t0 = Time.now
  ds = Dataset.new(split, vocab: vocab)

  #сколько реальных не <pad> токенов, сколько <unk>
  real = ds.ids.ne(Config::PAD_ID).count #сравнение сразу всей матрицы с нудем, получается маска 0 и 1
  unk = ds.ids.eq(Config::UNK_ID).count

  #считаем баланс классов
  balance = ds.labels.bincount.to_a

  puts "#{split}: чанков #{ds.size}, документов #{ds.doc_ids.uniq.size}, " \
         "классы #{balance}, <unk> #{(100.0 * unk / real).round(2)}%, " \
         "#{(Time.now - t0).round(1)}сек"
end