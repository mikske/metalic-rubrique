# frozen_string_literal: true

module Rubrique
  # Порт normalize_text_for_cnn + tokenize_text из диплома
  # Важно повторить его один в один, словарь и эмбеддинги построены на этой токенизации и любое расхождение даст лишний <unk>
  module Tokenizer
    WORD_RE = /[а-яёa-z0-9]+|[.,!?;:()"-]/i

    def self.normalize(text)
      text.downcase
          .tr("\n", " ")
          .gsub(/<[^>]+>/, " ") #убираем <CHAPTER> и прочие теги
          .gsub(/\s+/, " ")
          .strip
    end

    #re.findall = scan

    def self.tokenize(text)
      normalize(text).scan(WORD_RE)
    end
  end
end
