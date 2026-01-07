package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals;

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

    public Frame() {
        setTitle("Test-World: " + LadyBug.MAX_AGE / 3000 + "min");
        setLayout(new GridLayout());
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setVisible(true);
        this.game = new Game();
        setContentPane(game);
        pack();
        setResizable(false);
        gameLoop();
    }

    public void gameLoop() {
        Thread game = new Thread(() -> {
            final int fps = 50;
            final long frameTime = 1000 / fps;
            final String path = "src/net/forlindon/dynamic/objectoriented/neat/experiments/artifical/life/sim/test/out";

            int tick = 0;
            int minute = 0;

            while (true) {
                long start = System.currentTimeMillis();

                this.game.world.tickEverything();

                SwingUtilities.invokeLater(this.game::repaint);

                long sleep = frameTime - (System.currentTimeMillis() - start);
                if (sleep > 0) {
                    try {
                        Thread.sleep(sleep);
                    }
                    catch (InterruptedException e) {}
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
                tick %= 6000;
            }
        });
        game.setDaemon(true);
        game.start();
    }

    private void tick(ActionEvent e) {
        this.game.world.tickEverything();
        this.game.world.OBJECTS.sort(Object::compareTo);
        repaint();
    }
}
