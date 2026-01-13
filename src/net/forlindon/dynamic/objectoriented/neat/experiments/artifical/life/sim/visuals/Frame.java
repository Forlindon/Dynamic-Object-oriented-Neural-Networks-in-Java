package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals;

import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.neat.BufferEntry;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.utils.RingBuffer;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.objects.LadyBug;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.objects.Object;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Frame extends JFrame {

    Game game;
    RingBuffer<BufferEntry> BUFFER = new RingBuffer<>(2048);

    public Frame() {
        setTitle("Test-World: " + LadyBug.MAX_AGE / 1200 + "min");
        setLayout(new GridLayout());
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setVisible(true);
        this.game = new Game(this.BUFFER);
        setContentPane(game);
        pack();
        setResizable(false);
        startRepainting();
        startTicking();
        startCalculating();
    }

    public void startRepainting() {
        Thread repainting = new Thread(() -> {
            final int fps = 50;
            final long frameTime = 1000 / fps;

            while (true) {
                long start = System.currentTimeMillis();

                SwingUtilities.invokeLater(this.game::repaint);

                long sleep = frameTime - (System.currentTimeMillis() - start);
                if (sleep > 0) {
                    try {
                        Thread.sleep(sleep);
                    } catch (InterruptedException e) {}
                }
            }
        });
        repainting.setDaemon(true);
        repainting.start();
    }

    public void startTicking() {
        Thread ticking = new Thread(() -> {
            final int fps = 50;
            final long frameTime = 1000 / fps;
            final String path = "src/net/forlindon/dynamic/objectoriented/neat/experiments/artifical/life/sim/test/out";

            int tick = 0;
            int minute = 0;
            int ticksPerMinute = (int) (frameTime * 60);

            while (true) {
                long start = System.currentTimeMillis();

                this.game.world.tickEverything();

                long sleep = frameTime - (System.currentTimeMillis() - start);
                if (sleep > 0) {
                    try {
                        Thread.sleep(sleep);
                    } catch (InterruptedException e) {}
                }
                if (tick++ == 0) {
                    BufferedImage bi = new BufferedImage(this.game.getPreferredSize().width, this.game.getPreferredSize().height,BufferedImage.TYPE_INT_ARGB);
                    Graphics2D g = bi.createGraphics();
                    this.game.paintAll(g);
                    g.dispose();
                    System.out.println(path + "/" + minute + ".png");
                    try {
                        ImageIO.write(bi, "PNG", new File(path + "/" + minute + ".png"));
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                    System.out.printf("%d: %s - %s\n", minute++, this.getTitle(), this.game.world.speciesManager);
                    if (this.game.world.speciesManager.getPopulationSize() <= 1) {
                        System.exit(0);
                    }
                }
                tick %= ticksPerMinute;
            }
        });
        ticking.setDaemon(true);
        ticking.start();
    }

    public void  startCalculating() {
        Thread calculations = new Thread(() -> {

            while (!Thread.currentThread().isInterrupted()) {
                BufferEntry bufferEntry = this.BUFFER.pop();
                if (bufferEntry != null) {
                    bufferEntry.phenoType().forward(bufferEntry.input(), bufferEntry.out());
                }
                else {
                    Thread.onSpinWait();
                }
            }

        });
        calculations.setDaemon(false);
        calculations.start();
    }

    private void tick(ActionEvent e) {
        this.game.world.tickEverything();
        this.game.world.OBJECTS.sort(Object::compareTo);
        repaint();
    }
}
