/**
 * Adam with global gradient-norm clipping, operating on a list of flat
 * parameter tensors so the actor, the critic and the log-std vector can share
 * a single optimizer state.
 */
export class Adam {
  constructor(tensors, { learningRate, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 }) {
    this.tensors = tensors;
    this.learningRate = learningRate;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    this.states = tensors.map((tensor) => ({
      m: new Float32Array(tensor.params.length),
      v: new Float32Array(tensor.params.length),
    }));
    this.stepCount = 0;
  }

  zeroGrad() {
    for (const tensor of this.tensors) tensor.grads.fill(0);
  }

  scaleGradients(scale) {
    for (const tensor of this.tensors) {
      const grads = tensor.grads;
      for (let index = 0; index < grads.length; index += 1) {
        grads[index] *= scale;
      }
    }
  }

  gradientNorm() {
    let total = 0;
    for (const tensor of this.tensors) {
      const grads = tensor.grads;
      for (let index = 0; index < grads.length; index += 1) {
        total += grads[index] * grads[index];
      }
    }
    return Math.sqrt(total);
  }

  clipGradients(maxNorm) {
    const norm = this.gradientNorm();
    if (norm > maxNorm && norm > 0) {
      this.scaleGradients(maxNorm / norm);
    }
    return norm;
  }

  step() {
    this.stepCount += 1;
    const biasCorrection1 = 1 - Math.pow(this.beta1, this.stepCount);
    const biasCorrection2 = 1 - Math.pow(this.beta2, this.stepCount);
    for (let index = 0; index < this.tensors.length; index += 1) {
      const { params, grads } = this.tensors[index];
      const { m, v } = this.states[index];
      for (let offset = 0; offset < params.length; offset += 1) {
        const gradient = grads[offset];
        m[offset] = this.beta1 * m[offset] + (1 - this.beta1) * gradient;
        v[offset] = this.beta2 * v[offset] + (1 - this.beta2) * gradient * gradient;
        const mHat = m[offset] / biasCorrection1;
        const vHat = v[offset] / biasCorrection2;
        params[offset] -=
          (this.learningRate * mHat) / (Math.sqrt(vHat) + this.eps);
      }
    }
  }
}
