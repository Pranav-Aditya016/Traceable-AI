import { motion } from 'framer-motion';
import ParticleField from '../three/ParticleField';

export function Hero() {
  return (
    <div className="relative min-h-[80vh] flex items-center justify-center overflow-hidden" style={{ background: 'var(--hero-bg)' }}>
      <ParticleField />
      <div className="relative z-10 text-center px-4 max-w-3xl mx-auto">
        <motion.h1
          className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-white via-[var(--accent-light)] to-[var(--medical-cyan)] bg-clip-text text-transparent"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          AI that explains itself.
        </motion.h1>
        <motion.p
          className="text-lg md:text-xl text-[var(--text-secondary)] mb-10 leading-relaxed"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          Upload any medical document. TraceMD runs the full diagnostic pipeline .
        </motion.p>
        <motion.a
          href="#upload"
          className="inline-block px-8 py-4 bg-[var(--accent)] text-white font-semibold rounded-xl hover:bg-[var(--accent-light)] transition-colors shadow-lg shadow-[var(--accent)]/20"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          Upload Document →
        </motion.a>
      </div>
    </div>
  );
}
