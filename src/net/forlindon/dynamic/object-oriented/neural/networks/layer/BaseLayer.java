package layer;

import knot.Knot;

public class BaseLayer extends Layer {
    public BaseLayer(int id) {
        super(id);
    }

    @Override
    public void forward() {
        this.PARAMETERS.forEach(Knot::pop);
    }
}
