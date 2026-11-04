# frozen_string_literal: true
require "json"
require "numo/narray"
require_relative "config"
require_relative "tokenizer"
module Rubrique
  class Dataset
    MAX_SEQ_LEN = 700

    attr_reader :ids, :labels, :doc_ids, :size

    #split train, eval, text
    def initialize(split, vocab:, path: Config::CHUNKS_PATH, max_len: MAX_SEQ_LEN)
      rows = File.foreach(path) #читаем построчно, файл в память целиком не грузим
                 .map { |line| JSON.parse(line) }
                 .select { |row| row["split"] == split }

      @size = rows.size
      @labels = Numo::Int32.cast(rows.map { |r| r["label"] }) #[N]
      @doc_ids = rows.map { |r| r["doc_id"] } #для последующей агрегации по документам

      #Матрица индексов [N, max_len], <pad> = 0
      @ids = Numo::Int32.zeros(@size, max_len)
      rows.each_with_index do |row, i|
        tokens = Tokenizer.tokenize(row["text"]).first(max_len) #обрезка
        idx = tokens.map { |t| vocab.fetch(t, Config::UNK_ID) } #неизвестное слово UNK
        @ids[i, 0...idx.size] = idx unless idx.empty? #хвост остается паддингом
      end
    end
  end
end
