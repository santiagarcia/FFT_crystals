"""
GIF 7 – OTIS Mathematical Implementation for the Flow Rule
===========================================================
Step-by-step mathematical walkthrough of how the 5 Jacobian
derivatives are obtained via OTIS seeding in THIS specific problem.

Scenes:
  1. The 5 input variables and what we differentiate w.r.t.
  2. Step 1: Seeding — assign each variable its imaginary direction
  3. Step 2: Intermediate quantities (driving stress, resistances)
  4. Step 3: Stress ratio and the exponential
  5. Step 4: Final result — extract 5 derivatives from imaginary parts

Render:
    manim -ql gif_jacobian_otis.py JacobianOTIS
"""
from manim import *

C_BG     = WHITE
C_TEXT   = BLACK
C_BLUE   = "#1565C0"
C_RED    = "#D32F2F"
C_GREEN  = "#2E7D32"
C_ORANGE = "#E65100"
C_PURPLE = "#6A1B9A"
C_TEAL   = "#00695C"
C_GRAY   = "#444444"


class JacobianOTIS(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        # ==================================================================
        # SCENE 1 — The problem: flow rule and 5 derivatives needed
        # ==================================================================
        t1 = Text("Mathematical Implementation", font_size=42,
                  color=C_TEXT, weight=BOLD).to_edge(UP, buff=0.3)
        self.play(Write(t1), run_time=1.0)

        # Full flow rule
        fr = MathTex(
            r"\Delta\gamma^\alpha = \dot{\gamma}_0\,\Delta t\;"
            r"\exp\!\left[-\frac{\Delta G_0}{k_B T}"
            r"\left(1 - s^{\,p}\right)^{q}\right]"
            r"\;\mathrm{sgn}(\tau_d)",
            font_size=34, color=C_TEXT,
        )
        fr.next_to(t1, DOWN, buff=0.4)
        self.play(FadeIn(fr), run_time=1.0)
        self.wait(0.5)

        # Where s is
        where_s = MathTex(
            r"s = \frac{\tau_{\mathrm{eff}}}{r_{\mathrm{th}}}",
            r",\quad \tau_{\mathrm{eff}} = |\tau_d| - r_{\mathrm{ath}}",
            r",\quad \tau_d = \tau - \chi",
            font_size=30, color=C_GRAY,
        )
        where_s.next_to(fr, DOWN, buff=0.3)
        self.play(FadeIn(where_s), run_time=0.8)
        self.wait(0.5)

        # The 5 inputs with colors
        inp_label = Text("5 independent input variables:",
                         font_size=28, color=C_GRAY)
        inp_label.next_to(where_s, DOWN, buff=0.4)

        inp_data = [
            (r"\tau", "resolved stress", C_BLUE),
            (r"\chi", "backstress", C_RED),
            (r"r_{\mathrm{SSD}}", "thermal SSD", C_GREEN),
            (r"r_{\mathrm{CS}}", "cross-slip", C_ORANGE),
            (r"r_{\mathrm{ath,SSD}}", "athermal SSD", C_PURPLE),
        ]
        inp_grp = VGroup()
        for tex, desc, col in inp_data:
            m = MathTex(tex, font_size=32, color=col)
            t = Text(f" {desc}", font_size=22, color=C_GRAY)
            inp_grp.add(VGroup(m, t).arrange(RIGHT, buff=0.15))
        inp_grp.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        inp_grp.next_to(inp_label, DOWN, buff=0.25)

        self.play(FadeIn(inp_label), run_time=0.5)
        for row in inp_grp:
            self.play(FadeIn(row), run_time=0.3)
        self.wait(0.5)

        goal = MathTex(
            r"\text{Goal: compute }",
            r"\frac{\partial \Delta\gamma}{\partial \tau}",
            r",\;\frac{\partial \Delta\gamma}{\partial \chi}",
            r",\;\frac{\partial \Delta\gamma}{\partial r_{\mathrm{SSD}}}",
            r",\;\frac{\partial \Delta\gamma}{\partial r_{\mathrm{CS}}}",
            r",\;\frac{\partial \Delta\gamma}{\partial r_{\mathrm{ath,SSD}}}",
            font_size=28, color=C_TEXT,
        )
        goal[1].set_color(C_BLUE)
        goal[2].set_color(C_RED)
        goal[3].set_color(C_GREEN)
        goal[4].set_color(C_ORANGE)
        goal[5].set_color(C_PURPLE)
        goal.next_to(inp_grp, DOWN, buff=0.4)
        self.play(FadeIn(goal), run_time=1.0)
        self.wait(2.0)

        self.play(*[FadeOut(m) for m in
                    [t1, fr, where_s, inp_label, inp_grp, goal]],
                  run_time=0.6)

        # ==================================================================
        # SCENE 2a — Step 1: Seeding — concept
        # ==================================================================
        t2 = Text("Step 1: Seed Each Input", font_size=40,
                  color=C_TEXT, weight=BOLD).to_edge(UP, buff=0.3)
        self.play(Write(t2), run_time=0.8)

        # Explain the idea
        idea1 = Text("We need 5 derivatives, so we assign each input",
                      font_size=26, color=C_GRAY)
        idea2_txt = Text("its own imaginary direction ",
                         font_size=26, color=C_GRAY)
        idea2_eps = MathTex(r"\varepsilon_k", font_size=30, color=C_GRAY)
        idea2_colon = Text(" :", font_size=26, color=C_GRAY)
        idea2 = VGroup(idea2_txt, idea2_eps, idea2_colon).arrange(RIGHT, buff=0.08)
        idea_grp = VGroup(idea1, idea2).arrange(DOWN, buff=0.08)
        idea_grp.next_to(t2, DOWN, buff=0.35)
        self.play(FadeIn(idea_grp), run_time=0.7)
        self.wait(0.5)

        # General formula
        gen_eq = MathTex(
            r"\tilde{x}_k = x_k + 1\cdot\varepsilon_k",
            font_size=36, color=C_TEXT,
        )
        gen_eq.next_to(idea_grp, DOWN, buff=0.4)
        gen_box = SurroundingRectangle(gen_eq, color=C_TEAL, buff=0.15,
                                        corner_radius=0.1, stroke_width=2)
        self.play(FadeIn(gen_box), Write(gen_eq), run_time=1.0)
        self.wait(0.8)

        # Explain epsilon property
        eps_prop = MathTex(
            r"\varepsilon_i \,\varepsilon_j = 0 \;\;\forall\; i,j",
            font_size=30, color=C_GRAY,
        )
        eps_note_t1 = Text("Imaginary parts never mix \u2014 each ",
                           font_size=24, color=C_TEAL)
        eps_note_m = MathTex(r"\varepsilon_k", font_size=28, color=C_TEAL)
        eps_note_t2 = Text(" propagates independently.",
                           font_size=24, color=C_TEAL)
        eps_note = VGroup(eps_note_t1, eps_note_m, eps_note_t2).arrange(RIGHT, buff=0.08)
        eps_grp = VGroup(eps_prop, eps_note).arrange(DOWN, buff=0.15)
        eps_grp.next_to(gen_box, DOWN, buff=0.4)
        self.play(FadeIn(eps_grp), run_time=0.8)
        self.wait(2.0)

        self.play(*[FadeOut(m) for m in
                    [idea_grp, gen_eq, gen_box, eps_grp]], run_time=0.5)

        # ==================================================================
        # SCENE 2b — Step 1 cont.: the 5 specific seedings
        # ==================================================================
        seed_intro = Text("For our 5 inputs, the seeded variables are:",
                          font_size=26, color=C_GRAY)
        seed_intro.next_to(t2, DOWN, buff=0.35)
        self.play(FadeIn(seed_intro), run_time=0.5)

        seeds = [
            (r"\tilde{\tau}",                r"= \tau + 1\cdot \varepsilon_1",  C_BLUE),
            (r"\tilde{\chi}",                r"= \chi + 1\cdot \varepsilon_2",    C_RED),
            (r"\tilde{r}_{\mathrm{SSD}}",    r"= r_{\mathrm{SSD}} + 1\cdot \varepsilon_3", C_GREEN),
            (r"\tilde{r}_{\mathrm{CS}}",     r"= r_{\mathrm{CS}} + 1\cdot \varepsilon_4",  C_ORANGE),
            (r"\tilde{r}_{\mathrm{ath,SSD}}", r"= r_{\mathrm{ath,SSD}} + 1\cdot \varepsilon_5", C_PURPLE),
        ]
        seed_eqs = VGroup()
        for lhs, rhs, col in seeds:
            eq = MathTex(lhs, rhs, font_size=34, color=col)
            seed_eqs.add(eq)
        seed_eqs.arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        seed_eqs.next_to(seed_intro, DOWN, buff=0.3)

        for eq in seed_eqs:
            self.play(FadeIn(eq, shift=RIGHT*0.15), run_time=0.35)
        self.wait(0.8)

        note = Text("From here on, every arithmetic operation carries\n"
                     "all 5 imaginary parts through the computation.",
                     font_size=24, color=C_TEAL, line_spacing=1.2)
        note.next_to(seed_eqs, DOWN, buff=0.35)
        self.play(FadeIn(note), run_time=0.7)
        self.wait(2.5)

        self.play(*[FadeOut(m) for m in
                    [t2, seed_intro, seed_eqs, note]], run_time=0.6)

        # ==================================================================
        # SCENE 3a — Driving stress: step-by-step
        # ==================================================================
        t3 = Text("Step 2: Compute Intermediate Quantities",
                  font_size=38, color=C_TEXT, weight=BOLD).to_edge(UP, buff=0.3)
        self.play(Write(t3), run_time=0.8)

        # -- Driving stress --
        drv_label = Text("Driving stress", font_size=28, color=C_TEXT, weight=BOLD)
        drv_label.next_to(t3, DOWN, buff=0.35)
        self.play(FadeIn(drv_label), run_time=0.5)

        # Original formula
        drv_def = MathTex(
            r"\tilde{\tau}_d = \tilde{\tau} - \tilde{\chi}",
            font_size=34, color=C_TEXT,
        )
        drv_def.next_to(drv_label, DOWN, buff=0.3)
        self.play(Write(drv_def), run_time=0.8)
        self.wait(0.6)

        # Substitute the seeds
        drv_sub_label = Text("Substitute the seeded values:",
                             font_size=24, color=C_GRAY)
        drv_sub_label.next_to(drv_def, DOWN, buff=0.35)
        drv_sub = MathTex(
            r"= \bigl(\tau + \varepsilon_1\bigr)",
            r"- \bigl(\chi + \varepsilon_2\bigr)",
            font_size=32, color=C_TEXT,
        )
        drv_sub[0].set_color(C_BLUE)
        drv_sub[1].set_color(C_RED)
        drv_sub.next_to(drv_sub_label, DOWN, buff=0.2)
        self.play(FadeIn(drv_sub_label), run_time=0.4)
        self.play(FadeIn(drv_sub), run_time=0.8)
        self.wait(0.6)

        # Result — show symbolically, not numerical
        drv_result = MathTex(
            r"\tilde{\tau}_d",
            r"= \underbrace{(\tau - \chi)}_{\text{real part}}",
            r"+ \underbrace{\frac{\partial\tau_d}{\partial\tau}}_{= +1}",
            r"\varepsilon_1",
            r"+ \underbrace{\frac{\partial\tau_d}{\partial\chi}}_{= -1}",
            r"\varepsilon_2",
            font_size=28, color=C_TEXT,
        )
        drv_result[3].set_color(C_BLUE)
        drv_result[5].set_color(C_RED)
        drv_result.next_to(drv_sub, DOWN, buff=0.35)
        self.play(Write(drv_result), run_time=1.5)

        drv_note = Text("The coefficient of each direction is always the\n"
                        "partial derivative of the output w.r.t. that input.",
                        font_size=24, color=C_TEAL, line_spacing=1.2)
        drv_note.next_to(drv_result, DOWN, buff=0.3)
        self.play(FadeIn(drv_note), run_time=0.6)
        self.wait(2.5)

        self.play(*[FadeOut(m) for m in
                    [drv_label, drv_def, drv_sub_label, drv_sub,
                     drv_result, drv_note]], run_time=0.5)

        # ==================================================================
        # SCENE 3b — Athermal resistance: step-by-step
        # ==================================================================
        ath_label = Text("Athermal resistance", font_size=28, color=C_TEXT, weight=BOLD)
        ath_label.next_to(t3, DOWN, buff=0.35)
        self.play(FadeIn(ath_label), run_time=0.5)

        # Formula
        ath_def = MathTex(
            r"\tilde{r}_{\mathrm{ath}} = \sqrt{\tilde{r}_{\mathrm{ath,SSD}}^{\,2}"
            r"+ r_{\mathrm{ath,GND}}^2\,} + r_{\mathrm{ath},0}",
            font_size=30, color=C_TEXT,
        )
        ath_def.next_to(ath_label, DOWN, buff=0.3)
        self.play(Write(ath_def), run_time=0.8)
        self.wait(0.5)

        # Point out which terms are seeded
        ath_sub_label = MathTex(
            r"\text{Only } \tilde{r}_{\mathrm{ath,SSD}} "
            r"\text{ carries } \varepsilon_5 \text{; the rest are pure reals.}",
            font_size=26, color=C_PURPLE,
        )
        ath_sub_label.next_to(ath_def, DOWN, buff=0.3)
        self.play(FadeIn(ath_sub_label), run_time=0.7)
        self.wait(0.5)

        # Substitute
        ath_sub = MathTex(
            r"= \sqrt{(r_{\mathrm{ath,SSD}} + \varepsilon_5)^2"
            r"+ r_{\mathrm{ath,GND}}^2\,} + r_{\mathrm{ath},0}",
            font_size=28, color=C_TEXT,
        )
        ath_sub.next_to(ath_sub_label, DOWN, buff=0.25)
        self.play(FadeIn(ath_sub), run_time=0.8)
        self.wait(0.5)

        # Result
        ath_result = MathTex(
            r"\Rightarrow \tilde{r}_{\mathrm{ath}} = r_{\mathrm{ath}}"
            r"+ \frac{\partial r_{\mathrm{ath}}}{\partial r_{\mathrm{ath,SSD}}}\,"
            r"\varepsilon_5",
            font_size=28, color=C_PURPLE,
        )
        ath_note = Text("The overloaded sqrt automatically computes this coefficient.",
                        font_size=24, color=C_TEAL)
        ath_result_grp = VGroup(ath_result, ath_note).arrange(DOWN, buff=0.15)
        ath_result_grp.next_to(ath_sub, DOWN, buff=0.3)
        self.play(FadeIn(ath_result_grp), run_time=1.0)
        self.wait(2.5)

        self.play(*[FadeOut(m) for m in
                    [ath_label, ath_def, ath_sub_label, ath_sub,
                     ath_result_grp]], run_time=0.5)

        # ==================================================================
        # SCENE 3c — Thermal resistance: step-by-step
        # ==================================================================
        th_label = Text("Thermal resistance", font_size=28, color=C_TEXT, weight=BOLD)
        th_label.next_to(t3, DOWN, buff=0.35)
        self.play(FadeIn(th_label), run_time=0.5)

        # Formula
        th_def = MathTex(
            r"\tilde{r}_{\mathrm{th}} = \sqrt{\tilde{r}_{\mathrm{SSD}}^{\,2}"
            r"+ r_{\mathrm{GND}}^2"
            r"+ \tilde{r}_{\mathrm{CS}}^{\,2}\,}"
            r"+ r_{\mathrm{th},0}",
            font_size=28, color=C_TEXT,
        )
        th_def.next_to(th_label, DOWN, buff=0.3)
        self.play(Write(th_def), run_time=0.8)
        self.wait(0.5)

        # Which terms carry directions
        th_sub_label = MathTex(
            r"\tilde{r}_{\mathrm{SSD}} \text{ carries } \varepsilon_3"
            r",\quad \tilde{r}_{\mathrm{CS}} \text{ carries } \varepsilon_4",
            font_size=26, color=C_GRAY,
        )
        th_sub_label.next_to(th_def, DOWN, buff=0.3)
        self.play(FadeIn(th_sub_label), run_time=0.7)
        self.wait(0.5)

        # Substitute
        th_sub = MathTex(
            r"= \sqrt{(r_{\mathrm{SSD}} + \varepsilon_3)^2"
            r"+ r_{\mathrm{GND}}^2"
            r"+ (r_{\mathrm{CS}} + \varepsilon_4)^2\,}"
            r"+ r_{\mathrm{th},0}",
            font_size=26, color=C_TEXT,
        )
        th_sub.next_to(th_sub_label, DOWN, buff=0.25)
        self.play(FadeIn(th_sub), run_time=0.8)
        self.wait(0.5)

        # Result
        th_result = MathTex(
            r"\Rightarrow \tilde{r}_{\mathrm{th}} = r_{\mathrm{th}}"
            r"+ \frac{\partial r_{\mathrm{th}}}{\partial r_{\mathrm{SSD}}}\,"
            r"\varepsilon_3"
            r"+ \frac{\partial r_{\mathrm{th}}}{\partial r_{\mathrm{CS}}}\,"
            r"\varepsilon_4",
            font_size=26, color=C_TEXT,
        )
        th_note = Text("Two seeded inputs in the sqrt produce two derivative coefficients.",
                        font_size=24, color=C_TEAL)
        th_result_grp = VGroup(th_result, th_note).arrange(DOWN, buff=0.15)
        th_result_grp.next_to(th_sub, DOWN, buff=0.3)
        self.play(FadeIn(th_result_grp), run_time=1.0)
        self.wait(2.5)

        self.play(*[FadeOut(m) for m in
                    [t3, th_label, th_def, th_sub_label, th_sub,
                     th_result_grp]], run_time=0.6)

        # ==================================================================
        # SCENE 4a — Effective stress
        # ==================================================================
        t4 = Text("Step 3: Effective Stress and Ratio",
                  font_size=38, color=C_TEXT, weight=BOLD).to_edge(UP, buff=0.3)
        self.play(Write(t4), run_time=0.8)

        eff_label = Text("Effective stress", font_size=28, color=C_TEXT, weight=BOLD)
        eff_label.next_to(t4, DOWN, buff=0.35)
        self.play(FadeIn(eff_label), run_time=0.5)

        eff_def = MathTex(
            r"\tilde{\tau}_{\mathrm{eff}}",
            r"= |\tilde{\tau}_d| - \tilde{r}_{\mathrm{ath}}",
            font_size=34, color=C_TEXT,
        )
        eff_def.next_to(eff_label, DOWN, buff=0.3)
        self.play(Write(eff_def), run_time=0.8)
        self.wait(0.5)

        # Piecewise note
        pw_note = MathTex(
            r"|\tau_d| \text{ and } \mathrm{sgn}(\tau_d) \text{ are non-smooth at } \tau_d = 0.",
            font_size=24, color=C_RED,
        )
        pw_note.next_to(eff_def, DOWN, buff=0.25)
        pw_note2 = Text("OTI derivatives are valid in the active region where ",
                        font_size=22, color=C_GRAY)
        pw_note2_m = MathTex(r"\tau_{\mathrm{eff}} > 0", font_size=26, color=C_RED)
        pw_note2_end = Text(".", font_size=22, color=C_GRAY)
        pw_line = VGroup(pw_note2, pw_note2_m, pw_note2_end).arrange(RIGHT, buff=0.06)
        pw_grp = VGroup(pw_note, pw_line).arrange(DOWN, buff=0.1)
        pw_grp.next_to(eff_def, DOWN, buff=0.25)
        self.play(FadeIn(pw_grp), run_time=0.8)
        self.wait(1.5)

        eff_inherit = MathTex(
            r"\tilde{\tau}_d \text{ carries }",
            r"\varepsilon_1, \varepsilon_2",
            r"\quad\text{and}\quad",
            r"\tilde{r}_{\mathrm{ath}} \text{ carries }",
            r"\varepsilon_5",
            font_size=26, color=C_GRAY,
        )
        eff_inherit[1].set_color(C_BLUE)
        eff_inherit[4].set_color(C_PURPLE)
        eff_inherit.next_to(pw_grp, DOWN, buff=0.25)
        self.play(FadeIn(eff_inherit), run_time=0.7)
        self.wait(0.5)

        eff_result = MathTex(
            r"\Rightarrow \tilde{\tau}_{\mathrm{eff}} \text{ now carries }",
            r"\varepsilon_1",
            r",\;",
            r"\varepsilon_2",
            r",\;",
            r"\varepsilon_5",
            font_size=28, color=C_TEXT,
        )
        eff_result[1].set_color(C_BLUE)
        eff_result[3].set_color(C_RED)
        eff_result[5].set_color(C_PURPLE)
        eff_result.next_to(eff_inherit, DOWN, buff=0.25)
        self.play(FadeIn(eff_result), run_time=0.8)
        self.wait(2.0)

        self.play(*[FadeOut(m) for m in
                    [eff_label, eff_def, pw_grp, eff_inherit, eff_result]], run_time=0.5)

        # ==================================================================
        # SCENE 4b — Stress ratio (division merges all 5)
        # ==================================================================
        ratio_label = Text("Stress ratio", font_size=28, color=C_TEXT, weight=BOLD)
        ratio_label.next_to(t4, DOWN, buff=0.35)
        self.play(FadeIn(ratio_label), run_time=0.5)

        ratio_def = MathTex(
            r"\tilde{s} = \frac{\tilde{\tau}_{\mathrm{eff}}}{\tilde{r}_{\mathrm{th}}}",
            font_size=34, color=C_TEXT,
        )
        ratio_def.next_to(ratio_label, DOWN, buff=0.3)
        self.play(Write(ratio_def), run_time=0.8)
        self.wait(0.5)

        ratio_inherit = MathTex(
            r"\tilde{\tau}_{\mathrm{eff}} \text{ carries }",
            r"\varepsilon_1, \varepsilon_2, \varepsilon_5",
            r"\qquad",
            r"\tilde{r}_{\mathrm{th}} \text{ carries }",
            r"\varepsilon_3, \varepsilon_4",
            font_size=26, color=C_GRAY,
        )
        ratio_inherit.next_to(ratio_def, DOWN, buff=0.3)
        self.play(FadeIn(ratio_inherit), run_time=0.7)
        self.wait(0.5)

        ratio_note = Text("Division mixes numerator and denominator directions.",
                          font_size=24, color=C_TEAL)
        ratio_note.next_to(ratio_inherit, DOWN, buff=0.25)
        self.play(FadeIn(ratio_note), run_time=0.5)

        ratio_result = MathTex(
            r"\Rightarrow \tilde{s} \text{ carries all 5: }",
            r"\varepsilon_1",
            r",\;",
            r"\varepsilon_2",
            r",\;",
            r"\varepsilon_3",
            r",\;",
            r"\varepsilon_4",
            r",\;",
            r"\varepsilon_5",
            font_size=28, color=C_TEXT,
        )
        ratio_result[1].set_color(C_BLUE)
        ratio_result[3].set_color(C_RED)
        ratio_result[5].set_color(C_GREEN)
        ratio_result[7].set_color(C_ORANGE)
        ratio_result[9].set_color(C_PURPLE)
        ratio_result.next_to(ratio_note, DOWN, buff=0.3)
        self.play(FadeIn(ratio_result), run_time=0.8)
        self.wait(2.0)

        self.play(*[FadeOut(m) for m in
                    [ratio_label, ratio_def, ratio_inherit,
                     ratio_note, ratio_result]], run_time=0.5)

        self.play(FadeOut(t4), run_time=0.3)

        # ==================================================================
        # SCENE 4c — Barrier, exponential, final slip
        # ==================================================================
        t4b = Text("Step 4: Barrier + Exponential",
                   font_size=38, color=C_TEXT, weight=BOLD).to_edge(UP, buff=0.3)
        self.play(Write(t4b), run_time=0.8)

        barrier_label = Text("Barrier, exponential, and slip increment",
                             font_size=28, color=C_TEXT, weight=BOLD)
        barrier_label.next_to(t4b, DOWN, buff=0.35)
        self.play(FadeIn(barrier_label), run_time=0.5)

        barrier_eq = MathTex(
            r"\tilde{B} = \left(1 - \tilde{s}^{\,p}\right)^{q}",
            font_size=34, color=C_TEXT,
        )
        barrier_eq.next_to(barrier_label, DOWN, buff=0.3)
        self.play(Write(barrier_eq), run_time=0.8)

        barrier_note = Text("Power operations preserve all 5 directions from s.",
                            font_size=24, color=C_GRAY)
        barrier_note.next_to(barrier_eq, DOWN, buff=0.2)
        self.play(FadeIn(barrier_note), run_time=0.5)
        self.wait(0.8)

        exp_eq = MathTex(
            r"\tilde{E} = \exp\!\left(-\frac{\Delta G_0}{k_B T}\,\tilde{B}\right)",
            font_size=34, color=C_TEXT,
        )
        exp_eq.next_to(barrier_note, DOWN, buff=0.3)
        self.play(Write(exp_eq), run_time=0.8)

        exp_note = Text("Exponential of a multi-dual number: all 5 directions survive.",
                        font_size=24, color=C_GRAY)
        exp_note.next_to(exp_eq, DOWN, buff=0.2)
        self.play(FadeIn(exp_note), run_time=0.5)
        self.wait(0.8)

        # Final result
        final_eq = MathTex(
            r"\widetilde{\Delta\gamma}",
            r"= \dot{\gamma}_0\,\Delta t \;\tilde{E}\;\mathrm{sgn}(\tau_d)",
            font_size=34, color=C_TEAL,
        )
        final_box = SurroundingRectangle(final_eq, color=C_TEAL, buff=0.15,
                                          corner_radius=0.1, stroke_width=2)
        final_grp = VGroup(final_box, final_eq)
        final_grp.next_to(exp_note, DOWN, buff=0.35)
        self.play(FadeIn(final_grp), run_time=1.0)

        final_note = Text("Multiplying by constants and sign does not add new\n"
                          "directions, so all 5 derivative coefficients are preserved.",
                          font_size=24, color=C_GRAY, line_spacing=1.2)
        final_note.next_to(final_grp, DOWN, buff=0.3)
        self.play(FadeIn(final_note), run_time=0.8)
        self.wait(2.5)

        self.play(*[FadeOut(m) for m in
                    [t4b, barrier_label, barrier_eq, barrier_note,
                     exp_eq, exp_note, final_grp, final_note]], run_time=0.6)

        # ==================================================================
        # SCENE 5 — Step 5: Extract derivatives
        # ==================================================================
        t5 = Text("Step 5: Read Off the Derivatives",
                  font_size=40, color=C_TEXT, weight=BOLD).to_edge(UP, buff=0.3)
        self.play(Write(t5), run_time=0.8)

        result_eq = MathTex(
            r"\widetilde{\Delta\gamma}",
            r"= \underbrace{\Delta\gamma}_{\text{real part}}",
            r"+ \underbrace{\frac{\partial\Delta\gamma}{\partial\tau}}_{\varepsilon_1}\,\varepsilon_1",
            r"+ \underbrace{\frac{\partial\Delta\gamma}{\partial \chi}}_{\varepsilon_2}\,\varepsilon_2",
            r"+ \underbrace{\frac{\partial\Delta\gamma}{\partial r_{\mathrm{SSD}}}}_{\varepsilon_3}\,\varepsilon_3",
            r"+ \underbrace{\frac{\partial\Delta\gamma}{\partial r_{\mathrm{CS}}}}_{\varepsilon_4}\,\varepsilon_4",
            r"+ \underbrace{\frac{\partial\Delta\gamma}{\partial r_{\mathrm{ath}}}}_{\varepsilon_5}\,\varepsilon_5",
            font_size=22, color=C_TEXT,
        )
        result_eq[2].set_color(C_BLUE)
        result_eq[3].set_color(C_RED)
        result_eq[4].set_color(C_GREEN)
        result_eq[5].set_color(C_ORANGE)
        result_eq[6].set_color(C_PURPLE)
        result_eq.next_to(t5, DOWN, buff=0.3)
        self.play(Write(result_eq), run_time=2.5)
        self.wait(1.5)

        extract_label = Text("Extract each component:",
                              font_size=26, color=C_GRAY)
        extract_label.next_to(result_eq, DOWN, buff=0.3)
        self.play(FadeIn(extract_label), run_time=0.5)

        extractions = [
            (r"\widetilde{\Delta\gamma}\big|_{\varepsilon_1}",
             r"= \frac{\partial \Delta\gamma}{\partial \tau}", C_BLUE),
            (r"\widetilde{\Delta\gamma}\big|_{\varepsilon_2}",
             r"= \frac{\partial \Delta\gamma}{\partial \chi}", C_RED),
            (r"\widetilde{\Delta\gamma}\big|_{\varepsilon_3}",
             r"= \frac{\partial \Delta\gamma}{\partial r_{\mathrm{SSD}}}", C_GREEN),
            (r"\widetilde{\Delta\gamma}\big|_{\varepsilon_4}",
             r"= \frac{\partial \Delta\gamma}{\partial r_{\mathrm{CS}}}", C_ORANGE),
            (r"\widetilde{\Delta\gamma}\big|_{\varepsilon_5}",
             r"= \frac{\partial \Delta\gamma}{\partial r_{\mathrm{ath,SSD}}}", C_PURPLE),
        ]

        # Arrange in 2 columns to save vertical space
        left_col = VGroup()
        right_col = VGroup()
        for i, (lhs, rhs, col) in enumerate(extractions):
            eq = MathTex(lhs, rhs, font_size=26, color=col)
            if i < 3:
                left_col.add(eq)
            else:
                right_col.add(eq)
        left_col.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        right_col.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        cols = VGroup(left_col, right_col).arrange(RIGHT, buff=0.8)
        cols.next_to(extract_label, DOWN, buff=0.2)

        self.play(LaggedStart(*[FadeIn(eq, shift=RIGHT*0.1) 
                                for eq in [*left_col, *right_col]],
                              lag_ratio=0.2), run_time=2.0)
        self.wait(2.5)

        # Final summary
        summary = MathTex(
            r"\text{1 evaluation of } f",
            r"\;\longrightarrow\;",
            r"\Delta\gamma \text{ and all 5 } \frac{\partial\Delta\gamma}{\partial x_k}",
            font_size=28, color=C_TEAL,
        )
        summary_box = SurroundingRectangle(summary, color=C_TEAL, buff=0.15,
                                            corner_radius=0.1, stroke_width=2.5)
        s_grp = VGroup(summary_box, summary)
        s_grp.next_to(cols, DOWN, buff=0.2)

        # OTI scope clarification
        scope_line = MathTex(
            r"\text{OTI provides }\partial\Delta\gamma/\partial(\cdot)"
            r"\text{ at the material point; these feed the consistent tangent.}",
            font_size=20, color=C_GRAY,
        )
        scope_line.next_to(s_grp, DOWN, buff=0.12)

        self.play(FadeIn(s_grp), run_time=1.0)
        self.play(FadeIn(scope_line), run_time=0.8)
        self.wait(3.0)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)

        # ==================================================================
        # SCENE 6a — Bridge to code: Variables
        # ==================================================================
        t6 = Text("Bridge to Code — Variables", font_size=42,
                  color=C_TEXT, weight=BOLD).to_edge(UP, buff=0.3)
        self.play(Write(t6), run_time=0.8)

        var_maps = [
            (r"\tau",              "stressShearResolved(i)",     C_BLUE),
            (r"\chi",              "resistanceBackStressNew(i)", C_RED),
            (r"r_{\mathrm{ath}}",  "resistanceAthermalTotal(i)", C_PURPLE),
            (r"r_{\mathrm{th}}",   "resistanceThermalTotal(i)",  C_GREEN),
            (r"\Delta\gamma",      "incrementSlipPlastic(i)",    C_TEAL),
        ]
        var_rows = VGroup()
        for tex, code, col in var_maps:
            m = MathTex(tex, font_size=40, color=col)
            arrow = MathTex(r"\longrightarrow", font_size=36, color=C_GRAY)
            c = Text(code, font_size=26, color=C_TEXT, font="Consolas")
            row = VGroup(m, arrow, c).arrange(RIGHT, buff=0.25)
            var_rows.add(row)
        var_rows.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        var_rows.next_to(t6, DOWN, buff=0.4)
        var_rows.set_x(0)  # center horizontally

        for row in var_rows:
            self.play(FadeIn(row), run_time=0.3)
        self.wait(2.5)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)

        # ==================================================================
        # SCENE 6b — Bridge to code: Derivatives
        # ==================================================================
        t6b = Text("Bridge to Code — Derivatives", font_size=42,
                   color=C_TEXT, weight=BOLD).to_edge(UP, buff=0.3)
        self.play(Write(t6b), run_time=0.8)

        der_maps = [
            (r"\frac{\partial\Delta\gamma}{\partial\tau}",
             "dIncSlipPlastic_dStressResolved(i)", C_BLUE),
            (r"\frac{\partial\Delta\gamma}{\partial\chi}",
             "dIncSlipPlastic_dBackstress(i)", C_RED),
            (r"\frac{\partial\Delta\gamma}{\partial r_{\mathrm{ath}}}",
             "dIncSlipPlastic_dAthermalResist...(i)", C_PURPLE),
            (r"\frac{\partial\Delta\gamma}{\partial r_{\mathrm{th}}}",
             "dIncSlipPlastic_dThermalResist...(i)", C_GREEN),
        ]
        der_rows = VGroup()
        for tex, code, col in der_maps:
            m = MathTex(tex, font_size=36, color=col)
            arrow = MathTex(r"\longrightarrow", font_size=32, color=C_GRAY)
            c = Text(code, font_size=22, color=C_TEXT, font="Consolas")
            row = VGroup(m, arrow, c).arrange(RIGHT, buff=0.25)
            der_rows.add(row)
        der_rows.arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        der_rows.next_to(t6b, DOWN, buff=0.4)
        der_rows.set_x(0)  # center horizontally

        for row in der_rows:
            self.play(FadeIn(row), run_time=0.3)
        self.wait(3.0)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0)
