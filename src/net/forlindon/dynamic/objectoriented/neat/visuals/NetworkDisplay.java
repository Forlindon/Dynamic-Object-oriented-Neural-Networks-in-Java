package net.forlindon.dynamic.objectoriented.neat.visuals;

import net.forlindon.dynamic.objectoriented.neural.networks.connection.Connection;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;
import net.forlindon.dynamic.objectoriented.neural.networks.layer.Layer;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

import javax.swing.*;
import javax.swing.Timer;
import java.awt.*;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

public class NetworkDisplay extends JPanel {

    Layer l;

    public NetworkDisplay(Layer l) {
        this.l = l;

        Timer timer = new Timer(25, _ -> repaint());
        timer.start();
    }

    @Override
    public void paint(Graphics graph) {
        super.paint(graph);
        Graphics2D g = (Graphics2D) graph;
        List<Knot> knots = this.l.getKNOTS();
        List<Integer> ids = knots.stream().map(Knot::id).distinct().toList();

        int w = this.getWidth();
        int h = this.getHeight();

        g.setColor(Color.BLACK);
        g.fillRect(0,0,w,h);

        int x = (int)(w*0.1);
        int y = (int)(h*0.1);

        w=(int)(w*0.8);
        h=(int)(h*0.8);

        g.setColor(Color.GRAY);
        g.fillRect(x,y,w,h);

        int section = (int)(w*0.8)/ids.size();
        int gap = (int)(w*0.2)/ids.size();

        x+=gap/2;
        for (int i = 0; i < ids.size(); i++) {
            g.fillRect(x+i*(section+gap),y,section,h);
        }

        Map<Integer, Long> knotsPerLayer = knots.stream().collect(
                Collectors.groupingBy(
                        Knot::id,
                        Collectors.counting()
                )
        );

        int maxLayerSize = (int) knotsPerLayer.values().stream().mapToLong(Long::longValue).max().getAsLong();

        Map<Integer, Long> counter = knots.stream().map(Knot::id).distinct().collect(
                Collectors.toMap(
                        key -> key,
                        _ -> 0L
                )
        );

        int hSection = h/(maxLayerSize);
        int r = hSection/2;

        Map<Knot, Vec2d> positions = new HashMap<>();

        for (Knot k : knots) {
            int layerIdx = ids.indexOf(k.id());
            int count = counter.get(k.id()).intValue();

            int idx = maxLayerSize/2-(knotsPerLayer.get(k.id()).intValue()/2-count);

            int kX = x+(section+gap)*layerIdx+section/2-r/2;
            int kY = y+(hSection)*idx+r/2;

            positions.put(k,new Vec2d(kX,kY));

            counter.put(k.id(), counter.get(k.id())+1);
        }

        g.setColor(Color.BLACK);

        for (Tensor par : this.l.getParameters()) {
            if (par instanceof Connection c) {
                Vec2d src = positions.get(c.src());
                Vec2d dest = positions.get(c.dest());

                if (c.val == 0) continue;

                if (c.val < 0) g.setColor(Color.RED);
                else g.setColor(Color.BLACK);

                g.drawLine(src.x()+r/2,src.y()+r/2,dest.x()+r/2,dest.y()+r/2);
            }
        }

        g.setColor(Color.BLUE);
        for (Vec2d vec : positions.values()) {
            g.fillOval(vec.x(),vec.y,r,r);
        }

    }

    public synchronized void update() {
        this.repaint();
    }

    private record Vec2d(int x, int y) {
    }
}
