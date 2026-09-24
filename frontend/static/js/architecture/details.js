/**
 * Content of the explanation drawer of the "How the model works" page:
 * one entry per clickable element (`data-detail="<key>"`), with a title,
 * the source file it refers to in OpenFold3, and HTML (`.formula-block`
 * elements hold LaTeX rendered with KaTeX).
 */
export const DETAILS = {
  "pairformer-self-attn": {
    title: "Self-attention with pair bias",
    source: "layers/attention_pair_bias.py:121-227 · primitives/attention.py:94-135",
    html: `
      <p>This layer updates the <strong>single</strong> representation (one vector per residue) by letting every residue "look at" all the others. It works like classic self-attention, except that a bias computed from <strong>z</strong> (the pair representation) is added to the scores before the softmax. This is how the model conditions attention on what it already knows about the geometric relationships between residues.</p>
      <h4>The bias from z</h4>
      <div class="formula-block">$$b^{\\text{pair}} = \\text{permute}\\big(\\text{Linear}_z(\\text{LayerNorm}(z))\\big) \\in \\mathbb{R}^{H \\times N \\times N}$$</div>
      <h4>Full attention (per head h)</h4>
      <div class="formula-block">$$A_h = \\text{softmax}\\!\\left(\\frac{Q_h K_h^\\top}{\\sqrt{d}} + b^{\\text{pair}}_h + b^{\\text{mask}}\\right), \\qquad \\text{output} = A_h V_h$$</div>
      <p class="formula-note">The 1/√d scaling is applied to Q before the dot product (it is not visible in the softmax formula itself). The mask bias is −∞ for invalid positions and 0 otherwise.</p>
      <table class="dims-table">
        <tr><td>Residues (N)</td><td>sequence length</td></tr>
        <tr><td>Heads (H)</td><td>16</td></tr>
        <tr><td>Hidden size per head</td><td>24 (384 / 16)</td></tr>
        <tr><td>Repeated</td><td>48 times (once per Pairformer block)</td></tr>
      </table>
    `,
  },
  "triangle-mult": {
    title: "Triangle multiplication (outgoing / incoming)",
    source: "layers/triangular_multiplicative_update.py:421-503",
    html: `
      <p>Before any attention, each Pairformer block updates <strong>z</strong> with a purely multiplicative operation: no softmax and no attention weights, just two projections of z combined element-wise and summed over a third residue k. This is why this mechanism does not appear in the attention explorer.</p>
      <h4>Projections</h4>
      <div class="formula-block">$$a_{ik} = \\text{mask}_{ik} \\cdot \\sigma\\big(\\text{Linear}_g(z_{ik})\\big) \\odot \\text{Linear}_p(z_{ik}), \\qquad b_{jk} \\text{ (same, on } z_{jk}\\text{)}$$</div>
      <h4>Output, "outgoing" case</h4>
      <div class="formula-block">$$z_{ij}^{\\text{new}} = \\text{Linear}_z\\!\\left(\\text{LayerNorm}\\Big(\\sum_k a_{ik} \\odot b_{jk}\\Big)\\right) \\odot \\sigma\\big(\\text{Linear}_g(z_{ij})\\big)$$</div>
      <p class="formula-note">The "incoming" case applies the same formula with a and b transposed (indices swapped to (k, i) and (k, j)). The code uses exactly the same layers; only the axis permutation changes the direction of information flow in the triangle (i, j, k).</p>
      <table class="dims-table">
        <tr><td>Hidden size</td><td>128</td></tr>
        <tr><td>Repeated</td><td>2× per block (outgoing then incoming), × 48 blocks</td></tr>
      </table>
    `,
  },
  "triangle-start": {
    title: "Triangle attention · starting node",
    source: "layers/triangular_attention.py:31-172 (AF3 Algorithm 14)",
    html: `
      <p>Unlike self-attention on <strong>s</strong>, triangle attention works directly on <strong>z</strong> (the pair representation) and considers a third residue k for each pair (i, j): "for the pair (i, j), which residues k matter most for this relationship, starting from i?". This mechanism lets AlphaFold-style models respect three-body geometric constraints, such as the triangle inequality on distances.</p>
      <h4>Triangle bias</h4>
      <div class="formula-block">$$b_{h,i,j} = \\text{permute}\\big(\\text{Linear}_z(\\text{LayerNorm}(z))\\big)$$</div>
      <h4>Attention along the k axis, for each fixed i</h4>
      <div class="formula-block">$$A_{h,i,j,k} = \\text{softmax}_k\\!\\left(\\frac{Q_{h,i,j} K_{h,i,k}^\\top}{\\sqrt{d}} + b_{h,i,k}\\right)$$</div>
      <p class="formula-note">The explorer captures exactly this tensor A (averaged over heads) and shows it either as the full [i, j, k] cube, or averaged over the query index j into an [i, k] map. Averaging over k instead would give a constant 1/N map, since the softmax normalises over k.</p>
      <table class="dims-table">
        <tr><td>Heads</td><td>4</td></tr>
        <tr><td>Raw captured shape</td><td>N × N × N (a cube, not a matrix)</td></tr>
        <tr><td>Repeated</td><td>48 times</td></tr>
      </table>
    `,
  },
  "triangle-end": {
    title: "Triangle attention · ending node",
    source: "layers/triangular_attention.py:126-179 (AF3 Algorithm 15)",
    html: `
      <p>Exactly the same mechanism as the "starting node" version, applied to the <strong>transposed</strong> pair representation: the code calls <code>z.transpose(-2, -3)</code> before the computation and undoes it afterwards. Attention therefore runs along a fixed column j instead of a fixed row i; together, the two directions let the model reason about the triangle (i, j, k) both ways.</p>
      <div class="formula-block">$$\\text{TriangleAttentionEnding}(z) = \\text{TriangleAttentionStarting}(z^{\\top})^{\\top}$$</div>
      <p class="formula-note">The transpose applies to the two residue axes of z (indices i and j), not to the heads or channels.</p>
      <table class="dims-table">
        <tr><td>Heads</td><td>4</td></tr>
        <tr><td>Raw captured shape</td><td>N × N × N (a cube, not a matrix)</td></tr>
        <tr><td>Repeated</td><td>48 times</td></tr>
      </table>
    `,
  },
  "diffusion-schedule": {
    title: "Diffusion · noise schedule and denoising steps",
    source: "structure/diffusion_module.py:82-119, 278-400",
    html: `
      <p>The diffusion module starts from a purely random 3D point cloud and progressively denoises it, over about 200 steps, into the final coordinates. At each step the network (including the <strong>diffusion transformer</strong> analysed here) predicts the denoised cloud, and the sampler takes a small step towards that prediction.</p>
      <h4>Noise schedule σ(t), EDM style</h4>
      <div class="formula-block">$$\\sigma(t) = \\sigma_{\\text{data}} \\cdot \\Big(s_{\\max}^{1/p} + t \\cdot (s_{\\min}^{1/p} - s_{\\max}^{1/p})\\Big)^{p}, \\qquad t \\in \\{0, \\tfrac{1}{T}, \\ldots, 1\\}$$</div>
      <p class="formula-note">Defaults: σ_data = 16, s_max = 160, s_min = 4×10⁻⁴, p = 7.</p>
      <h4>One denoising step</h4>
      <div class="formula-block">$$\\delta = \\frac{x_{\\text{noisy}} - x_{\\text{denoised}}}{t}, \\qquad x_{\\tau} = x_{\\text{noisy}} + \\text{step\\_scale} \\cdot (c_\\tau - t) \\cdot \\delta$$</div>
      <p class="formula-note">step_scale = 1.5, gamma_0 = 0.8, gamma_min = 1.0, noise_scale = 1.003. A comment by the authors in the code flags a deliberate deviation from the original EDM formula (x_noisy is used instead of x at this step).</p>
      <table class="dims-table">
        <tr><td>Diffusion transformer</td><td>24 blocks · 16 heads</td></tr>
        <tr><td>Rollout steps</td><td>~200 (only one, mid-rollout, is captured)</td></tr>
      </table>
    `,
  },
  "msa-averaging": {
    title: "MSA pair-weighted averaging",
    source: "layers/msa.py:560-624 (AF3 Algorithm 10)",
    html: `
      <p>In AlphaFold2, the MSA module used real self-attention between homologous sequences (row/column attention). AlphaFold3 simplifies it: the weights of the average over sequences no longer come from a Query·Key product but directly from a projection of <strong>z</strong>. This module therefore has no attention map in the usual sense and does not appear in the explorer.</p>
      <h4>Weights, independent of the MSA itself</h4>
      <div class="formula-block">$$w_{ij}^h = \\text{softmax}_j\\Big(\\text{Linear}_z(\\text{LayerNorm}(z))_{i,j,h}\\Big)$$</div>
      <h4>Weighted average</h4>
      <div class="formula-block">$$\\text{out}_i^h = \\sum_j w_{ij}^h \\cdot \\text{Linear}_v(m_j)^h$$</div>
      <p class="formula-note">m is the MSA representation (one row per homologous sequence). Since w only depends on z, every sequence of the MSA is averaged with exactly the same weights.</p>
      <table class="dims-table">
        <tr><td>Repeated</td><td>4 blocks (MSAModule)</td></tr>
      </table>
    `,
  },
  "adaln": {
    title: "AdaLN · normalisation conditioned on s",
    source: "primitives/normalization.py:89-140 (AF3 Algorithm 26)",
    html: `
      <p>The diffusion transformer processes a signal <strong>a</strong> (the point cloud being denoised) that must stay aware of each residue's context, carried by <strong>s</strong>. Instead of a plain LayerNorm, the model uses adaptive normalisation: the scale and shift applied to <strong>a</strong> are themselves computed from <strong>s</strong>.</p>
      <div class="formula-block">$$\\hat a = \\text{LayerNorm}_{\\text{no affine}}(a), \\qquad \\hat s = \\text{LayerNorm}_{\\text{scale only}}(s)$$</div>
      <div class="formula-block">$$\\text{AdaLN}(a, s) = \\sigma\\big(\\text{Linear}_g(\\hat s)\\big) \\odot \\hat a + \\text{Linear}_s(\\hat s)$$</div>
      <p class="formula-note">A second modulation ("AdaLN-Zero") is applied after the attention itself: the output is multiplied again by σ(Linear(s)), so that each block can learn to change nothing at the start of training.</p>
    `,
  },
  "plddt": {
    title: "pLDDT · per-residue confidence",
    source: "core/metrics/confidence.py:18-46",
    html: `
      <p>The pLDDT shown on each structure (0 to 100) is not predicted directly: the model predicts a <strong>probability distribution over 50 confidence bins</strong>, and the score is the expected value of that distribution.</p>
      <div class="formula-block">$$p_b = \\text{softmax}(\\text{logits})_b, \\quad b = 0..49 \\qquad c_b = \\frac{b + 0.5}{50}$$</div>
      <div class="formula-block">$$\\text{pLDDT} = \\sum_{b=0}^{49} p_b \\cdot c_b \\;\\times\\; 100$$</div>
      <p class="formula-note">The same method (expected value over bins) is used for the PAE and PDE confidence metrics, with bins expressed in ångströms instead of normalised confidence.</p>
    `,
  },
};
