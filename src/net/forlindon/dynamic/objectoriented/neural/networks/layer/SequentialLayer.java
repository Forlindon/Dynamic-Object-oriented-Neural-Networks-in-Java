package net.forlindon.dynamic.objectoriented.neural.networks.layer;

import net.forlindon.dynamic.objectoriented.neural.networks.connection.Connection;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

public class SequentialLayer extends Layer {

    protected final List<Layer> LAYERS;

    @SafeVarargs
    public SequentialLayer(Function<Integer, Layer>... factories) {
        super(-1);
        this.LAYERS = new ArrayList<>();
        for (int i = 0; i < factories.length; i++) {
            this.LAYERS.add(factories[i].apply(i));
            this.PARAMETERS.addAll(LAYERS.get(i).PARAMETERS);
        }
    }

    protected SequentialLayer() {
        super(-1);
        this.LAYERS = new ArrayList<>();
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

    public void fullConnect(BiFunction<Knot, Knot, Connection> factory) {
        for (int i = 0; i < this.LAYERS.size()-1; i++) {
            this.LAYERS.get(i).fullConnect(this.LAYERS.get(i+1), factory);
        }
    }

    @Override
    public void fullConnect(Layer other, BiFunction<Knot, Knot, Connection> factory) {
        this.LAYERS.get(this.LAYERS.size()-1).fullConnect(other,factory);
    }

    @Override
    public void forward(double[] inputs) {
        this.LAYERS.get(0).forward(inputs);
        for (int i = 1; i < this.LAYERS.size(); i++) {
            this.LAYERS.get(i).forward();
        }
    }

    @Override
    public void backward() {
        this.LAYERS.get(this.LAYERS.size()-1).PARAMETERS.forEach(k -> k.pushGrad(1));
        for (int i = this.LAYERS.size()-1; i >= 0; i--) {
            Layer l = this.LAYERS.get(i);
            l.backward();
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

    @Override
    public List<Knot> getParameters() {
        List<Knot> params = new ArrayList<>();
        for (Layer l : this.LAYERS) {
            params.addAll(l.getParameters());
        }
        return params;
    }
}
