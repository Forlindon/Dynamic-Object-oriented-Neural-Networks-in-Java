package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals;

import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.neat.BufferEntry;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.utils.RingBuffer;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.tiles.BasicTile;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.world.World;

import javax.swing.*;
import java.awt.*;

public class Game extends JPanel {

    World world;

    public Game(RingBuffer<BufferEntry> BUFFER) {
        setPreferredSize(new Dimension(640,640));
        int w = getPreferredSize().width;
        int h = getPreferredSize().height;
        this.world = new World(BUFFER, w / BasicTile.TILE_SIZE,h / BasicTile.TILE_SIZE);
    }

    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;
        try {
            this.world.draw(g2d);
        } catch (Exception e) {
            System.out.println(e.getCause());
            throw new RuntimeException(e);
        }
    }
}
