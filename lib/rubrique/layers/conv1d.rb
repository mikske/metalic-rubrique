require 'numo/narray'

module Rubrique
  module Layers
    class Conv1D
      #пока что веса ставим руками для теста, позже попробуем подгрузить
      attr_accessor :weight, :bias
      attr_reader :grad_weight, :grad_bias, :kernel, :filters

      def initialize(in_dim:, filters:, kernel:)
        @in_dim, @filters, @kernel = in_dim, filters, kernel
        bound = 1.0 / Math.sqrt(kernel * in_dim)
        @weight = Numo::SFloat.new(kernel * in_dim, filters).rand(-bound, bound)
        @bias = Numo::SFloat.new(filters).rand(-bound, bound)
      end

      # forward
      def forward(x)

        #На шаге j в j-й блок каждой строки копируется слово  t+j,
        # сразу для всех t одним слайсом x[true, j...(j + 1), true]
        # После k шагов в каждой строке лежит окно из k слов подряд
        # bias формы добавляется к каждой строке через broadcasting

        b, t, e = x.shape
        raise ArgumentError, "ожидали Е=#{@in_dim}, пришло #{e}" unless e == @in_dim
        l = t - @kernel + 1
        raise ArgumentError, "T=#{t} короче ядра k=#{@kernel}" if l < 1

        cols = Numo::SFloat.zeros(b, l, @kernel * e)
        @kernel.times do |j|
          cols[true, true, (j * e)...((j + 1) * e)] = x[true, j...(j + l), true]
        end

        #im2col, цикл не по позициям t, а по сдвину j

        @cols = cols.reshape(b * l, @kernel * e)
        y = @cols.dot(@weight) + @bias
        y.reshape(b, l, filters)
      end

      def backward(dout)
        raise "backward до forward" unless @cols
        dy = dout.dup.reshape(@cols.shape[0], filters)
        @grad_weight = @cols.transpose.dot(dy)
        @grad_bias = dy.sum(axis: 0)
        nil
      end
    end
  end
end