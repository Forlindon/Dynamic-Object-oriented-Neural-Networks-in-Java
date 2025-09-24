package layer;

import connection.Connection;
import knot.Knot;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public class SequentialLayer extends Layer {

    private final List<Layer> LAYERS;

    @SafeVarargs
    public SequentialLayer(Function<Integer, Layer>... factories) {
        super(-1);
        this.LAYERS = new ArrayList<>();
        for (int i = 0; i < factories.length; i++) {
            this.LAYERS.add(factories[i].apply(i));
            this.PARAMETERS.addAll(LAYERS.get(i).PARAMETERS);
        }
    }

    public SequentialLayer(Layer... l) {
        super(-1);
        this.LAYERS = new ArrayList<>();
        this.LAYERS.addAll(Arrays.asList(l));
    }

    @Override
    public void forward() {
        for (Layer l : this.LAYERS) {
            l.forward();
        }
    }

    public void fullConnect(Function<Knot, Connection> factory) {
        for (int i = 0; i < this.LAYERS.size()-1; i++) {
            this.LAYERS.get(i).fullConnect(this.LAYERS.get(i+1), factory);
        }
    }

    @Override
    public void fullConnect(Layer other, Function<Knot, Connection> factory) {
        fullConnect(factory);
    }

    @Override
    public void forward(double[] inputs) {
        this.LAYERS.get(0).forward(inputs);
        for (int i = 1; i < this.LAYERS.size(); i++) {
            this.LAYERS.get(i).forward();
        }
    }

    @Override
    public void readValues(double[] vals) {
        this.LAYERS.get(this.LAYERS.size()-1).readValues(vals);
    }

    public Layer get(int idx) {
        return this.LAYERS.get(idx);
    }

    @Override
    public String toString() {
        return String.format("{%s}",this.LAYERS);
    }
}
