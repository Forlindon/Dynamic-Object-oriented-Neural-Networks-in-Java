package net.forlindon.dynamic.objectoriented.neat.visuals;

import net.forlindon.dynamic.objectoriented.neural.networks.layer.Layer;

public class NetWrapper {

    public Layer net;

    public NetWrapper(Layer l) {
        this.net = l;
    }

    public void set(Layer net) {
        this.net = net;
    }

    public Layer get() {
        return this.net;
    }

}
