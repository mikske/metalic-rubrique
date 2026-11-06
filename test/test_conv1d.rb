require "minitest/autorun"
require_relative "../lib/rubrique/layers/conv1d"

class Conv1DTest < Minitest::Test
  def setup
    Numo::NArray.srand(42)
  end

  def tiny_conv
    conv = Rubrique::Layers::Conv1D.new(in_dim: 2, filters: 1, kernel: 2)
    conv.weight = Numo::SFloat[[1], [0], [0], [1]]
    conv.bias = Numo::SFloat[0]
    conv
  end

  def tiny_x
    Numo::SFloat[[[1, 2], [3, 4], [5, 6], [7, 8]]]
  end

  def test_output_shape
    conv = Rubrique::Layers::Conv1D.new(in_dim: 300, filters: 128, kernel: 3)
    x = Numo::SFloat.new(2, 10, 300).rand
    assert_equal [2, 8, 128], conv.forward(x).shape
  end

  def test_tiny_example_forward
    y = tiny_conv.forward(tiny_x)
    assert_equal [1, 3, 1], y.shape
    assert_equal [5.0, 9.0, 13.0], y.flatten.to_a
  end

  def test_bias_is_added
    conv = tiny_conv
    conv.bias = Numo::SFloat[10]
    assert_equal [15.0, 19.0, 23.0], conv.forward(tiny_x).flatten.to_a
  end

  def test_tiny_example_backward
    conv = tiny_conv
    conv.forward(tiny_x)
    conv.backward(Numo::SFloat.ones(1, 3, 1))

    assert_equal [4, 1], conv.grad_weight.shape
    assert_equal [9.0, 12.0, 15.0, 18.0], conv.grad_weight.flatten.to_a
    assert_equal [3.0], conv.grad_bias.to_a
  end

  def test_backward_shapes
    conv = Rubrique::Layers::Conv1D.new(in_dim: 300, filters: 128, kernel: 5)
    y = conv.forward(Numo::SFloat.new(2, 10, 300).rand)
    conv.backward(Numo::SFloat.ones(*y.shape))
    assert_equal [1500, 128], conv.grad_weight.shape
    assert_equal [128], conv.grad_bias.shape
  end

  def test_too_short_input
    conv = Rubrique::Layers::Conv1D.new(in_dim: 2, filters: 1, kernel: 3)
    assert_raises(ArgumentError) { conv.forward(Numo::SFloat.new(1, 2, 2).rand) }
  end

  def test_wrong_embedding_dim
    conv = Rubrique::Layers::Conv1D.new(in_dim: 300, filters: 4, kernel: 3)
    assert_raises(ArgumentError) { conv.forward(Numo::SFloat.new(1, 10, 299).rand) }
  end
end
