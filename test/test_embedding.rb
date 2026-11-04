require "minitest/autorun"
require_relative '../lib/rubrique/layers/embedding'

class TestEmbedding < Minitest::Test
  def setup
    w = Numo::SFloat[[0.0, 0.0], [0.1, 0.2], [1.0, 0.5], [0.3, 0.9]]
    @layer = Rubrique::Layers::Embedding.new(w)
  end

  def test_shape
    out = @layer.forward(Numo::Int32[[2, 3, 0], [1, 1, 2]])
    assert_equal [2, 3, 2], out.shape #b=2, t=3, d=2
  end

  def test_lookup
    out = @layer.forward(Numo::Int32[[2, 3, 0]])
    assert_equal [1.0, 0.5], out[0, 0, true].to_a #гном
    assert_equal [0.0, 0.0], out[0, 2, true].to_a #<pad>
  end

  def test_frozen_backward
    assert_nil @layer.backward(Numo::SFloat.zeros(1, 3, 2))
  end
end