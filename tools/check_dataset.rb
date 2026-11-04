require 'json'
require_relative '../lib/rubrique/config'
require_relative '../lib/rubrique/tokenizer'
include Rubrique

mismatch = 0
total = 0
File.foreach(Config::CHUNKS_PATH).each do |line|
  row = JSON.parse(line)
  n = Tokenizer.tokenize(row["text"]).size
  total += 1
  mismatch += 1 if n != row["chunk_tokens"]
end
puts "Проверено чанков: #{total}, расхождение с Питоном: #{mismatch}"
