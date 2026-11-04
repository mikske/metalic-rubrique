require "numo/narray"

module Rubrique
  module Layers
    # Таблица векторов слов. Прямой проход
    class Embedding
      attr_reader :weights

      # weights: Numo::SFloat [V, D] - например, fastText из Embeddings.load
      def initialize(weights, frozen: true)
        @weights = weights
        @frozen = frozen
      end

      # ids: Numo::Int32 [В, Т]
      def forward(ids)
        b, t = ids.shape
        d = @weights.shape[1]

        # Numo вытиягивает строки только по одномерному массиву индексов, поэтому достаем [b,t] в плоский список длины b*t
        # потом берем строки [b*t, d] и возрващем форму обратно
        @weights[ids.flatten, true].reshape(b, t, d)
      end

      #grad: dL/dOut [b, t, d]
      def backward(grad)
        return nil if @frozen # замороженные эмбеддинги, градиент дальше не идет
        raise NotImplementedError, "trainable embeddings - вариант В, позже"
      end
    end
  end
end