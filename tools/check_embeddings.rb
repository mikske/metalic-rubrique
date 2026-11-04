# frozen_string_literal: true

require_relative "../lib/rubrique/config"
require_relative "../lib/rubrique/embeddings"

include Rubrique

emb = Embeddings.load(Config::EMB_PATH)
vocab = JSON.parse(File.read(Config::VOCAB_PATH))

puts "shape: #{emb.shape.inspect}, vocab: #{vocab.size}"
raise "Словарь и матрица разного размера" if emb.shape[0] != vocab.size

#норма вектора сумма квадратов, у <пад> должно быть 0

norm = ->(i) { Math.sqrt((emb[i, true] ** 2).sum) }

puts "pad norm: #{norm.(Config::PAD_ID)}"
puts "unk norm: #{norm.(Config::UNK_ID)}"

%w[в и на когда было гномы].each do |tok|
  idx = vocab[tok]
  puts "#{tok} idx: #{idx} norm: #{norm.(idx).round(6)}"
end