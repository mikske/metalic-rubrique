# frozen_string_literal: true
require "numo/narray"
require "json"

module Rubrique
  module Embeddings
    # Читаем сырой f32-файл, экспортированный из пайторч.
    # В файле только V*D подтряд, поэтому форму берем из соседнего json

    def self.load(path)
      meta = JSON.parse(File.read("#{path}.json"))
      rows, cols = meta["rows"], meta["cols"]

      raw = File.binread(path)

      #размер файла должен быть rows*cols*4 байта
      expected = rows * cols * 4
      raise "Размер не совпал: #{raw.bytesize} != #{expected}" if raw.bytesize != expected

      Numo::SFloat.from_binary(raw).reshape(rows, cols)
    end
  end
end
