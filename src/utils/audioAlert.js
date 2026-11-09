class AudioAlertManager {
  constructor(cooldownMs = 2500) {
    this.audioCtx = null;
    this.cooldownMs = cooldownMs;
    this.lastAlertTime = 0;
    this.isMuted = false;
  }

  init() {
    if (!this.audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass) {
        this.audioCtx = new AudioContextClass();
      }
    }
    if (this.audioCtx && this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }
  }

  setMuted(muted) {
    this.isMuted = muted;
  }

  setCooldown(cooldownMs) {
    this.cooldownMs = Math.max(500, cooldownMs);
  }

  canPlayAlert() {
    if (this.isMuted) return false;
    const now = Date.now();
    return (now - this.lastAlertTime) >= this.cooldownMs;
  }

  playHazardBeep(force = false) {
    if (this.isMuted && !force) return false;
    const now = Date.now();

    if (!force && (now - this.lastAlertTime < this.cooldownMs)) {
      return false;
    }

    try {
      this.init();
      if (!this.audioCtx) return false;

      const t = this.audioCtx.currentTime;

      const osc1 = this.audioCtx.createOscillator();
      const gain1 = this.audioCtx.createGain();

      osc1.type = 'sine';
      osc1.frequency.setValueAtTime(880, t);
      osc1.frequency.exponentialRampToValueAtTime(750, t + 0.12);

      gain1.gain.setValueAtTime(0.001, t);
      gain1.gain.exponentialRampToValueAtTime(0.35, t + 0.02);
      gain1.gain.exponentialRampToValueAtTime(0.001, t + 0.14);

      osc1.connect(gain1);
      gain1.connect(this.audioCtx.destination);

      osc1.start(t);
      osc1.stop(t + 0.15);

      const osc2 = this.audioCtx.createOscillator();
      const gain2 = this.audioCtx.createGain();

      osc2.type = 'triangle';
      osc2.frequency.setValueAtTime(660, t + 0.15);
      osc2.frequency.exponentialRampToValueAtTime(550, t + 0.30);

      gain2.gain.setValueAtTime(0.001, t + 0.15);
      gain2.gain.exponentialRampToValueAtTime(0.40, t + 0.17);
      gain2.gain.exponentialRampToValueAtTime(0.001, t + 0.32);

      osc2.connect(gain2);
      gain2.connect(this.audioCtx.destination);

      osc2.start(t + 0.15);
      osc2.stop(t + 0.33);

      this.lastAlertTime = now;
      return true;
    } catch (err) {
      console.warn('Audio playback error:', err);
      return false;
    }
  }
}

export const audioAlert = new AudioAlertManager(2500);
