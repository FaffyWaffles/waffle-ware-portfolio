// ═══════════════════════════════════════════════════════════════════════════════
// EquationWeb — Comprehensive Physical Equation Database
// ═══════════════════════════════════════════════════════════════════════════════
// This database maps every major physical equation across all fields of science
// and reveals hidden connections through shared variables and fundamental
// constants of the universe.
// ═══════════════════════════════════════════════════════════════════════════════

// ─── Field Metadata ──────────────────────────────────────────────────────────

const FIELD_INFO = {
    classical_mechanics:    { name: "Classical Mechanics",     color: "#4A90D9", icon: "⚙" },      // blue        H≈213°
    electromagnetism:       { name: "Electromagnetism",        color: "#E74C3C", icon: "⚡" },      // red         H≈6°
    thermodynamics:         { name: "Thermodynamics",          color: "#F39C12", icon: "🔥" },      // orange      H≈37°
    quantum_mechanics:      { name: "Quantum Mechanics",       color: "#7C4DFF", icon: "⚛" },      // blue-violet H≈259°
    special_relativity:     { name: "Special Relativity",      color: "#1ABC9C", icon: "🚀" },      // teal        H≈168°
    general_relativity:     { name: "General Relativity",      color: "#5C6BC0", icon: "🌌" },      // indigo      H≈231°
    chemistry:              { name: "Chemistry",               color: "#2ECC71", icon: "🧪" },      // green       H≈145°
    statistical_mechanics:  { name: "Statistical Mechanics",   color: "#A8D84E", icon: "📊" },      // lime        H≈81°
    fluid_mechanics:        { name: "Fluid Mechanics",         color: "#4FC3F7", icon: "🌊" },      // sky blue    H≈199°
    optics_waves:           { name: "Optics & Waves",          color: "#F1C40F", icon: "🔬" },      // yellow      H≈49°
    nuclear_physics:        { name: "Nuclear & Particle",      color: "#E91E63", icon: "☢" },      // pink        H≈340°
    astrophysics:           { name: "Astrophysics & Cosmology",color: "#D946EF", icon: "🌟" },      // orchid      H≈292°
    information_theory:     { name: "Information Theory",      color: "#00BCD4", icon: "💡" },      // cyan        H≈187°
    meta:                   { name: "Derived Constants",       color: "#95A5A6", icon: "🔗" },      // gray
};

// ─── Fundamental Constants of the Universe ───────────────────────────────────

const CONSTANTS = [
    {
        id: "const_c", name: "Speed of Light", symbol: "c",
        value: "2.99792458 × 10⁸", unit: "m/s",
        description: "Maximum speed of causality and information propagation in spacetime"
    },
    {
        id: "const_h", name: "Planck Constant", symbol: "h",
        value: "6.62607015 × 10⁻³⁴", unit: "J·s",
        description: "Fundamental quantum of action; relates photon energy to frequency"
    },
    {
        id: "const_hbar", name: "Reduced Planck Constant", symbol: "ℏ",
        value: "1.054571817 × 10⁻³⁴", unit: "J·s",
        description: "h/(2π) — natural unit of angular momentum in quantum mechanics"
    },
    {
        id: "const_G", name: "Gravitational Constant", symbol: "G",
        value: "6.67430 × 10⁻¹¹", unit: "N·m²/kg²",
        description: "Fundamental coupling constant of gravitation"
    },
    {
        id: "const_kB", name: "Boltzmann Constant", symbol: "k_B",
        value: "1.380649 × 10⁻²³", unit: "J/K",
        description: "Relates thermodynamic temperature to microscopic energy scale"
    },
    {
        id: "const_e", name: "Elementary Charge", symbol: "e",
        value: "1.602176634 × 10⁻¹⁹", unit: "C",
        description: "Magnitude of electric charge carried by a single proton"
    },
    {
        id: "const_NA", name: "Avogadro Constant", symbol: "N_A",
        value: "6.02214076 × 10²³", unit: "mol⁻¹",
        description: "Number of constituent particles per mole of substance"
    },
    {
        id: "const_R", name: "Gas Constant", symbol: "R",
        value: "8.314462618", unit: "J/(mol·K)",
        description: "Energy per temperature per mole; bridges macroscopic and molecular thermodynamics (R = N_A·k_B)"
    },
    {
        id: "const_eps0", name: "Vacuum Permittivity", symbol: "ε₀",
        value: "8.8541878128 × 10⁻¹²", unit: "F/m",
        description: "Permittivity of free space; governs electric field strength in vacuum"
    },
    {
        id: "const_mu0", name: "Vacuum Permeability", symbol: "μ₀",
        value: "1.25663706212 × 10⁻⁶", unit: "H/m",
        description: "Permeability of free space; governs magnetic field strength in vacuum"
    },
    {
        id: "const_sigma", name: "Stefan-Boltzmann Constant", symbol: "σ",
        value: "5.670374419 × 10⁻⁸", unit: "W/(m²·K⁴)",
        description: "Total radiant power per unit area of a blackbody; σ = 2π⁵k_B⁴/(15h³c²)"
    },
    {
        id: "const_ke", name: "Coulomb Constant", symbol: "k_e",
        value: "8.9875517923 × 10⁹", unit: "N·m²/C²",
        description: "Electrostatic force constant; k_e = 1/(4πε₀)"
    },
    {
        id: "const_me", name: "Electron Mass", symbol: "m_e",
        value: "9.1093837015 × 10⁻³¹", unit: "kg",
        description: "Rest mass of an electron"
    },
    {
        id: "const_mp", name: "Proton Mass", symbol: "m_p",
        value: "1.67262192369 × 10⁻²⁷", unit: "kg",
        description: "Rest mass of a proton"
    },
    {
        id: "const_alpha", name: "Fine-Structure Constant", symbol: "α",
        value: "≈ 1/137.036", unit: "dimensionless",
        description: "Fundamental coupling constant of QED; measures electromagnetic interaction strength"
    },
    {
        id: "const_a0", name: "Bohr Radius", symbol: "a₀",
        value: "5.29177210903 × 10⁻¹¹", unit: "m",
        description: "Most probable electron-nucleus distance in ground-state hydrogen"
    },
    {
        id: "const_Rinf", name: "Rydberg Constant", symbol: "R_∞",
        value: "1.0973731568160 × 10⁷", unit: "m⁻¹",
        description: "Relates to wavelengths of spectral lines of hydrogen-like atoms"
    },
    {
        id: "const_F", name: "Faraday Constant", symbol: "F",
        value: "96485.33212", unit: "C/mol",
        description: "Charge per mole of electrons; F = N_A·e"
    },
    {
        id: "const_bwien", name: "Wien Displacement Constant", symbol: "b",
        value: "2.897771955 × 10⁻³", unit: "m·K",
        description: "Relates blackbody peak emission wavelength to temperature"
    },
    {
        id: "const_g", name: "Standard Gravity", symbol: "g",
        value: "9.80665", unit: "m/s²",
        description: "Standard gravitational acceleration at Earth's surface"
    },
];

// ─── Physical Variables (Quantities) ─────────────────────────────────────────

const VARIABLES = [
    // Mechanical
    { id: "force",               name: "Force",                   symbol: "F",     unit: "N" },
    { id: "mass",                name: "Mass",                    symbol: "m",     unit: "kg" },
    { id: "acceleration",        name: "Acceleration",            symbol: "a",     unit: "m/s²" },
    { id: "velocity",            name: "Velocity",                symbol: "v",     unit: "m/s" },
    { id: "displacement",        name: "Displacement",            symbol: "x",     unit: "m" },
    { id: "distance",            name: "Distance / Separation",   symbol: "r",     unit: "m" },
    { id: "time",                name: "Time",                    symbol: "t",     unit: "s" },
    { id: "energy",              name: "Energy",                  symbol: "E",     unit: "J" },
    { id: "momentum",            name: "Momentum",                symbol: "p",     unit: "kg·m/s" },
    { id: "angular_momentum",    name: "Angular Momentum",        symbol: "L",     unit: "kg·m²/s" },
    { id: "moment_of_inertia",   name: "Moment of Inertia",      symbol: "I",     unit: "kg·m²" },
    { id: "angular_velocity",    name: "Angular Velocity",        symbol: "ω",     unit: "rad/s" },
    { id: "angular_acceleration",name: "Angular Acceleration",    symbol: "α",     unit: "rad/s²" },
    { id: "torque",              name: "Torque",                  symbol: "τ",     unit: "N·m" },
    { id: "spring_constant",     name: "Spring Constant",         symbol: "k",     unit: "N/m" },
    { id: "period",              name: "Period",                  symbol: "T",     unit: "s" },
    { id: "height",              name: "Height",                  symbol: "h",     unit: "m" },
    { id: "length",              name: "Length / Extent",         symbol: "L",     unit: "m" },
    { id: "radius",              name: "Radius",                  symbol: "R",     unit: "m" },
    { id: "work",                name: "Work",                    symbol: "W",     unit: "J" },
    { id: "power",               name: "Power",                   symbol: "P",     unit: "W" },
    { id: "angle",               name: "Angle",                   symbol: "θ",     unit: "rad" },
    // Wave & Optical
    { id: "frequency",           name: "Frequency",               symbol: "ν",     unit: "Hz" },
    { id: "wavelength",          name: "Wavelength",              symbol: "λ",     unit: "m" },
    { id: "intensity",           name: "Intensity",               symbol: "I",     unit: "W/m²" },
    { id: "refractive_index",    name: "Refractive Index",        symbol: "n",     unit: "" },
    { id: "spectral_radiance",   name: "Spectral Radiance",       symbol: "B",     unit: "W/(m²·sr·Hz)" },
    // Thermodynamic
    { id: "temperature",         name: "Temperature",             symbol: "T",     unit: "K" },
    { id: "pressure",            name: "Pressure",                symbol: "P",     unit: "Pa" },
    { id: "volume",              name: "Volume",                  symbol: "V",     unit: "m³" },
    { id: "entropy",             name: "Entropy",                 symbol: "S",     unit: "J/K" },
    { id: "heat",                name: "Heat",                    symbol: "Q",     unit: "J" },
    { id: "internal_energy",     name: "Internal Energy",         symbol: "U",     unit: "J" },
    { id: "enthalpy",            name: "Enthalpy",                symbol: "H",     unit: "J" },
    { id: "gibbs_energy",        name: "Gibbs Free Energy",       symbol: "G",     unit: "J" },
    { id: "helmholtz_energy",    name: "Helmholtz Free Energy",   symbol: "F",     unit: "J" },
    { id: "specific_heat",       name: "Specific Heat Capacity",  symbol: "c_p",   unit: "J/(kg·K)" },
    { id: "thermal_conductivity",name: "Thermal Conductivity",    symbol: "κ",     unit: "W/(m·K)" },
    { id: "efficiency",          name: "Efficiency",              symbol: "η",     unit: "" },
    { id: "num_microstates",     name: "Number of Microstates",   symbol: "Ω",     unit: "" },
    { id: "degrees_of_freedom",  name: "Degrees of Freedom",      symbol: "f",     unit: "" },
    { id: "moles",               name: "Amount of Substance",     symbol: "n",     unit: "mol" },
    // Electromagnetic
    { id: "charge",              name: "Electric Charge",         symbol: "q",     unit: "C" },
    { id: "electric_field",      name: "Electric Field",          symbol: "E",     unit: "V/m" },
    { id: "magnetic_field",      name: "Magnetic Field",          symbol: "B",     unit: "T" },
    { id: "electric_potential",   name: "Electric Potential",      symbol: "V",     unit: "V" },
    { id: "current",             name: "Electric Current",        symbol: "I",     unit: "A" },
    { id: "resistance",          name: "Resistance",              symbol: "R",     unit: "Ω" },
    { id: "capacitance",         name: "Capacitance",             symbol: "C",     unit: "F" },
    { id: "inductance",          name: "Inductance",              symbol: "L",     unit: "H" },
    { id: "magnetic_flux",       name: "Magnetic Flux",           symbol: "Φ_B",   unit: "Wb" },
    { id: "emf",                 name: "Electromotive Force",     symbol: "ε",     unit: "V" },
    { id: "area",                name: "Area",                    symbol: "A",     unit: "m²" },
    // Fluid
    { id: "density",             name: "Density",                 symbol: "ρ",     unit: "kg/m³" },
    { id: "viscosity",           name: "Dynamic Viscosity",       symbol: "μ",     unit: "Pa·s" },
    { id: "flow_rate",           name: "Volumetric Flow Rate",    symbol: "Q",     unit: "m³/s" },
    { id: "var_reynolds",         name: "Reynolds Number",         symbol: "Re",    unit: "" },
    // Quantum
    { id: "wave_function",       name: "Wave Function",           symbol: "ψ",     unit: "" },
    { id: "work_function",       name: "Work Function",           symbol: "φ",     unit: "J" },
    // Statistical
    { id: "var_partition_func",  name: "Partition Function",      symbol: "Z",     unit: "" },
    { id: "probability",         name: "Probability",             symbol: "P",     unit: "" },
    { id: "chemical_potential",  name: "Chemical Potential",       symbol: "μ",     unit: "J" },
    { id: "num_particles",       name: "Number of Particles",     symbol: "N",     unit: "" },
    // Nuclear
    { id: "decay_constant",      name: "Decay Constant",          symbol: "λ",     unit: "s⁻¹" },
    { id: "half_life",           name: "Half-Life",               symbol: "t₁/₂",  unit: "s" },
    // Chemistry
    { id: "concentration",       name: "Concentration",           symbol: "c",     unit: "mol/L" },
    { id: "activation_energy",   name: "Activation Energy",       symbol: "E_a",   unit: "J/mol" },
    { id: "equilibrium_constant",name: "Equilibrium Constant",    symbol: "K",     unit: "" },
    { id: "reaction_quotient",   name: "Reaction Quotient",       symbol: "Q",     unit: "" },
    { id: "absorbance",          name: "Absorbance",              symbol: "A",     unit: "" },
    { id: "molar_absorptivity",  name: "Molar Absorptivity",     symbol: "ε",     unit: "L/(mol·cm)" },
    { id: "path_length",         name: "Path Length",             symbol: "l",     unit: "cm" },
    // Relativistic
    { id: "lorentz_factor",      name: "Lorentz Factor",          symbol: "γ",     unit: "" },
    // Cosmological
    { id: "hubble_constant",     name: "Hubble Parameter",        symbol: "H₀",   unit: "km/s/Mpc" },
    { id: "luminosity",          name: "Luminosity",              symbol: "L",     unit: "W" },
    { id: "scale_factor",        name: "Scale Factor",            symbol: "a",     unit: "" },
    { id: "cosmological_constant",name: "Cosmological Constant",  symbol: "Λ",     unit: "m⁻²" },
    { id: "stress_energy_tensor",name: "Stress-Energy Tensor",    symbol: "T_μν",  unit: "" },
    { id: "metric_tensor",       name: "Metric Tensor",           symbol: "g_μν",  unit: "" },
    { id: "einstein_tensor",     name: "Einstein Tensor",         symbol: "G_μν",  unit: "" },
    // Information
    { id: "information_entropy", name: "Information Entropy",     symbol: "H",     unit: "bits" },
    // Extended variables
    { id: "cross_section",       name: "Cross-Section",           symbol: "σ",     unit: "m²" },
    { id: "magnetic_moment",     name: "Magnetic Moment",         symbol: "μ",     unit: "A·m²" },
    { id: "diffusion_coefficient",name: "Diffusion Coefficient",  symbol: "D",     unit: "m²/s" },
    { id: "surface_tension",     name: "Surface Tension",         symbol: "γ",     unit: "N/m" },
    { id: "atomic_number",       name: "Atomic Number",           symbol: "Z",     unit: "" },
    { id: "mass_number",         name: "Mass Number",             symbol: "A",     unit: "" },
    { id: "molar_mass",          name: "Molar Mass",              symbol: "M",     unit: "kg/mol" },
    { id: "electric_dipole",     name: "Electric Dipole Moment",  symbol: "p",     unit: "C·m" },
    { id: "bandwidth",           name: "Bandwidth",               symbol: "B",     unit: "Hz" },
    { id: "signal_noise_ratio",  name: "Signal-to-Noise Ratio",   symbol: "S/N",   unit: "" },
    { id: "angular_frequency",   name: "Angular Frequency",       symbol: "ω",     unit: "rad/s" },
    { id: "heat_capacity_ratio", name: "Heat Capacity Ratio",     symbol: "γ",     unit: "" },
    { id: "mean_free_path",      name: "Mean Free Path",          symbol: "λ_mfp", unit: "m" },
    { id: "spin_quantum",        name: "Spin Quantum Number",     symbol: "s",     unit: "" },
    { id: "orbital_quantum",     name: "Orbital Quantum Number",  symbol: "l",     unit: "" },
    { id: "principal_quantum",   name: "Principal Quantum Number",symbol: "n",     unit: "" },
    { id: "magnetic_quantum",    name: "Magnetic Quantum Number", symbol: "m_l",   unit: "" },
    { id: "solid_angle",         name: "Solid Angle",             symbol: "Ω",     unit: "sr" },
    { id: "proper_time",         name: "Proper Time",             symbol: "τ",     unit: "s" },
    { id: "impact_parameter",    name: "Impact Parameter",        symbol: "b",     unit: "m" },
    { id: "scattering_angle",    name: "Scattering Angle",        symbol: "θ",     unit: "rad" },
    { id: "compressibility",     name: "Compressibility Factor",  symbol: "Z_c",   unit: "" },
    { id: "strain",              name: "Strain",                  symbol: "ε",     unit: "" },
    { id: "stress",              name: "Stress",                  symbol: "σ",     unit: "Pa" },
    { id: "youngs_modulus",      name: "Young's Modulus",         symbol: "E",     unit: "Pa" },
    { id: "poisson_ratio",       name: "Poisson's Ratio",        symbol: "ν",     unit: "" },
    { id: "bulk_modulus",        name: "Bulk Modulus",            symbol: "K",     unit: "Pa" },
    { id: "shear_modulus",       name: "Shear Modulus",           symbol: "G",     unit: "Pa" },
    { id: "eccentricity",        name: "Orbital Eccentricity",   symbol: "e",     unit: "" },
    { id: "semi_major_axis",     name: "Semi-Major Axis",        symbol: "a",     unit: "m" },
    { id: "redshift",            name: "Redshift",               symbol: "z",     unit: "" },
    { id: "optical_depth",       name: "Optical Depth",          symbol: "τ",     unit: "" },
    { id: "electric_susceptibility", name: "Electric Susceptibility", symbol: "χ_e", unit: "" },
    { id: "polarization",        name: "Polarization Density",   symbol: "P",     unit: "C/m²" },
    { id: "magnetization",       name: "Magnetization",          symbol: "M",     unit: "A/m" },
    { id: "number_density",      name: "Number Density",         symbol: "n",     unit: "m⁻³" },
    { id: "current_density",     name: "Current Density",        symbol: "J",     unit: "A/m²" },
    { id: "resistivity_var",     name: "Resistivity",            symbol: "ρ",     unit: "Ω·m" },
    { id: "channel_capacity",    name: "Channel Capacity",       symbol: "C",     unit: "bits/s" },
    { id: "rate_constant",       name: "Rate Constant",          symbol: "k",     unit: "varies" },
    { id: "order_parameter",     name: "Order Parameter",        symbol: "φ",     unit: "" },
    { id: "correlation_length",  name: "Correlation Length",      symbol: "ξ",     unit: "m" },
    { id: "critical_exponent",   name: "Critical Exponent",      symbol: "β",     unit: "" },
];

// ─── Equations Database ──────────────────────────────────────────────────────
// Each equation references the variables and constants it contains.
// These references create the edges in the graph — the web of connections.

const EQUATIONS = [

    // ═════════════════════════════════════════════════════════════════════════
    // CLASSICAL MECHANICS
    // ═════════════════════════════════════════════════════════════════════════

    {
        id: "newton_second_law",
        name: "Newton's Second Law",
        field: "classical_mechanics",
        equation: "\\vec{F} = m\\vec{a}",
        description: "The net force on a body equals its mass times its acceleration. Foundation of classical dynamics.",
        uses: ["force", "mass", "acceleration"],
        year: 1687, discoverer: "Isaac Newton"
    },
    {
        id: "kinetic_energy",
        name: "Kinetic Energy",
        field: "classical_mechanics",
        equation: "E_k = \\tfrac{1}{2}mv^2",
        description: "Energy possessed by a body due to its motion.",
        uses: ["energy", "mass", "velocity"],
        year: 1829, discoverer: "Gaspard-Gustave de Coriolis"
    },
    {
        id: "gravitational_pe_surface",
        name: "Gravitational PE (Surface)",
        field: "classical_mechanics",
        equation: "U = mgh",
        description: "Potential energy near Earth's surface due to gravity.",
        uses: ["energy", "mass", "const_g", "height"],
        year: 1687, discoverer: "Isaac Newton"
    },
    {
        id: "newton_gravitation",
        name: "Newton's Law of Gravitation",
        field: "classical_mechanics",
        equation: "F = \\frac{GMm}{r^2}",
        description: "Every mass attracts every other mass with a force proportional to the product of their masses and inversely proportional to the square of their separation.",
        uses: ["force", "const_G", "mass", "distance"],
        year: 1687, discoverer: "Isaac Newton"
    },
    {
        id: "gravitational_pe_general",
        name: "Gravitational PE (General)",
        field: "classical_mechanics",
        equation: "U = -\\frac{GMm}{r}",
        description: "General gravitational potential energy between two masses.",
        uses: ["energy", "const_G", "mass", "distance"],
        year: 1687, discoverer: "Isaac Newton"
    },
    {
        id: "momentum_def",
        name: "Linear Momentum",
        field: "classical_mechanics",
        equation: "\\vec{p} = m\\vec{v}",
        description: "Momentum is mass times velocity — a conserved quantity in isolated systems.",
        uses: ["momentum", "mass", "velocity"],
        year: 1687, discoverer: "Isaac Newton"
    },
    {
        id: "centripetal_force",
        name: "Centripetal Force",
        field: "classical_mechanics",
        equation: "F = \\frac{mv^2}{r}",
        description: "Force required for circular motion, directed toward the center.",
        uses: ["force", "mass", "velocity", "distance"],
        year: 1659, discoverer: "Christiaan Huygens"
    },
    {
        id: "angular_momentum_def",
        name: "Angular Momentum",
        field: "classical_mechanics",
        equation: "L = I\\omega",
        description: "Rotational analog of linear momentum; conserved in absence of external torque.",
        uses: ["angular_momentum", "moment_of_inertia", "angular_velocity"],
        year: 1736, discoverer: "Leonhard Euler"
    },
    {
        id: "torque_def",
        name: "Torque",
        field: "classical_mechanics",
        equation: "\\tau = I\\alpha",
        description: "Rotational analog of Newton's second law.",
        uses: ["torque", "moment_of_inertia", "angular_acceleration"],
        year: 1736, discoverer: "Leonhard Euler"
    },
    {
        id: "rotational_ke",
        name: "Rotational Kinetic Energy",
        field: "classical_mechanics",
        equation: "E_{rot} = \\tfrac{1}{2}I\\omega^2",
        description: "Kinetic energy of a rotating body.",
        uses: ["energy", "moment_of_inertia", "angular_velocity"],
        year: 1736, discoverer: "Leonhard Euler"
    },
    {
        id: "hookes_law",
        name: "Hooke's Law",
        field: "classical_mechanics",
        equation: "F = -kx",
        description: "Restoring force of an ideal spring is proportional to displacement.",
        uses: ["force", "spring_constant", "displacement"],
        year: 1660, discoverer: "Robert Hooke"
    },
    {
        id: "shm_spring_period",
        name: "SHM Period (Spring)",
        field: "classical_mechanics",
        equation: "T = 2\\pi\\sqrt{\\frac{m}{k}}",
        description: "Period of a mass-spring oscillator.",
        uses: ["period", "mass", "spring_constant"],
        year: 1660, discoverer: "Robert Hooke"
    },
    {
        id: "shm_pendulum_period",
        name: "SHM Period (Pendulum)",
        field: "classical_mechanics",
        equation: "T = 2\\pi\\sqrt{\\frac{\\ell}{g}}",
        description: "Period of a simple pendulum for small oscillations.",
        uses: ["period", "length", "const_g"],
        year: 1673, discoverer: "Christiaan Huygens"
    },
    {
        id: "escape_velocity",
        name: "Escape Velocity",
        field: "classical_mechanics",
        equation: "v_e = \\sqrt{\\frac{2GM}{r}}",
        description: "Minimum speed to escape a gravitational field without further propulsion.",
        uses: ["velocity", "const_G", "mass", "distance"],
        year: 1728, discoverer: "Isaac Newton"
    },
    {
        id: "kepler_third_law",
        name: "Kepler's Third Law",
        field: "classical_mechanics",
        equation: "T^2 = \\frac{4\\pi^2}{GM}a^3",
        description: "The square of the orbital period is proportional to the cube of the semi-major axis.",
        uses: ["period", "const_G", "mass", "semi_major_axis"],
        year: 1619, discoverer: "Johannes Kepler"
    },
    {
        id: "work_def",
        name: "Work",
        field: "classical_mechanics",
        equation: "W = Fd\\cos\\theta",
        description: "Work done by a force over a displacement.",
        uses: ["work", "force", "distance", "angle"],
        year: 1829, discoverer: "Gaspard-Gustave de Coriolis"
    },
    {
        id: "power_mechanical",
        name: "Mechanical Power",
        field: "classical_mechanics",
        equation: "P = Fv",
        description: "Rate of doing work, or force times velocity.",
        uses: ["power", "force", "velocity"],
        year: 1782, discoverer: "James Watt"
    },
    {
        id: "orbital_velocity",
        name: "Orbital Velocity",
        field: "classical_mechanics",
        equation: "v = \\sqrt{\\frac{GM}{r}}",
        description: "Speed for a stable circular orbit around mass M at radius r.",
        uses: ["velocity", "const_G", "mass", "distance"],
        year: 1687, discoverer: "Isaac Newton"
    },

    // ═════════════════════════════════════════════════════════════════════════
    // ELECTROMAGNETISM
    // ═════════════════════════════════════════════════════════════════════════

    {
        id: "coulombs_law",
        name: "Coulomb's Law",
        field: "electromagnetism",
        equation: "F = k_e\\frac{q_1 q_2}{r^2}",
        description: "Electrostatic force between two point charges.",
        uses: ["force", "const_ke", "charge", "distance"],
        year: 1785, discoverer: "Charles-Augustin de Coulomb"
    },
    {
        id: "electric_field_point",
        name: "Electric Field (Point Charge)",
        field: "electromagnetism",
        equation: "\\vec{E} = k_e\\frac{Q}{r^2}\\hat{r}",
        description: "Electric field generated by a point charge.",
        uses: ["electric_field", "const_ke", "charge", "distance"],
        year: 1785, discoverer: "Charles-Augustin de Coulomb"
    },
    {
        id: "gauss_law",
        name: "Gauss's Law",
        field: "electromagnetism",
        equation: "\\oint \\vec{E} \\cdot d\\vec{A} = \\frac{Q_{enc}}{\\varepsilon_0}",
        description: "Total electric flux through a closed surface equals enclosed charge divided by ε₀. First of Maxwell's equations.",
        uses: ["electric_field", "area", "charge", "const_eps0"],
        year: 1835, discoverer: "Carl Friedrich Gauss"
    },
    {
        id: "electric_potential_point",
        name: "Electric Potential (Point Charge)",
        field: "electromagnetism",
        equation: "V = k_e\\frac{Q}{r}",
        description: "Scalar electric potential at distance r from a point charge.",
        uses: ["electric_potential", "const_ke", "charge", "distance"],
        year: 1785, discoverer: "Charles-Augustin de Coulomb"
    },
    {
        id: "coulomb_pe",
        name: "Electrostatic Potential Energy",
        field: "electromagnetism",
        equation: "U = k_e\\frac{q_1 q_2}{r}",
        description: "Potential energy stored in the configuration of two point charges.",
        uses: ["energy", "const_ke", "charge", "distance"],
        year: 1785, discoverer: "Charles-Augustin de Coulomb"
    },
    {
        id: "parallel_plate_capacitance",
        name: "Parallel Plate Capacitance",
        field: "electromagnetism",
        equation: "C = \\varepsilon_0 \\frac{A}{d}",
        description: "Capacitance of a parallel plate capacitor.",
        uses: ["capacitance", "const_eps0", "area", "distance"],
        year: 1745, discoverer: "Ewald Georg von Kleist"
    },
    {
        id: "capacitor_energy",
        name: "Capacitor Energy",
        field: "electromagnetism",
        equation: "U = \\tfrac{1}{2}CV^2",
        description: "Energy stored in a charged capacitor.",
        uses: ["energy", "capacitance", "electric_potential"],
        year: 1745, discoverer: "Ewald Georg von Kleist"
    },
    {
        id: "ohms_law",
        name: "Ohm's Law",
        field: "electromagnetism",
        equation: "V = IR",
        description: "Voltage across a conductor is proportional to the current through it.",
        uses: ["electric_potential", "current", "resistance"],
        year: 1827, discoverer: "Georg Simon Ohm"
    },
    {
        id: "electric_power",
        name: "Electric Power",
        field: "electromagnetism",
        equation: "P = IV",
        description: "Power dissipated or delivered in an electrical circuit.",
        uses: ["power", "current", "electric_potential"],
        year: 1841, discoverer: "James Prescott Joule"
    },
    {
        id: "biot_savart",
        name: "Biot-Savart Law (Long Wire)",
        field: "electromagnetism",
        equation: "B = \\frac{\\mu_0 I}{2\\pi r}",
        description: "Magnetic field around an infinitely long straight current-carrying wire.",
        uses: ["magnetic_field", "const_mu0", "current", "distance"],
        year: 1820, discoverer: "Jean-Baptiste Biot & Félix Savart"
    },
    {
        id: "faradays_law",
        name: "Faraday's Law of Induction",
        field: "electromagnetism",
        equation: "\\mathcal{E} = -\\frac{d\\Phi_B}{dt}",
        description: "Induced EMF equals negative rate of change of magnetic flux. Third of Maxwell's equations.",
        uses: ["emf", "magnetic_flux", "time"],
        year: 1831, discoverer: "Michael Faraday"
    },
    {
        id: "lorentz_force",
        name: "Lorentz Force Law",
        field: "electromagnetism",
        equation: "\\vec{F} = q(\\vec{E} + \\vec{v} \\times \\vec{B})",
        description: "Total electromagnetic force on a charged particle.",
        uses: ["force", "charge", "electric_field", "velocity", "magnetic_field"],
        year: 1895, discoverer: "Hendrik Lorentz"
    },
    {
        id: "em_wave_speed",
        name: "Speed of Light from EM Constants",
        field: "electromagnetism",
        equation: "c = \\frac{1}{\\sqrt{\\varepsilon_0 \\mu_0}}",
        description: "Speed of light emerges from electric and magnetic constants — revealing light as an electromagnetic wave.",
        uses: ["const_c", "const_eps0", "const_mu0"],
        year: 1865, discoverer: "James Clerk Maxwell"
    },
    {
        id: "magnetic_force_wire",
        name: "Force on Current-Carrying Wire",
        field: "electromagnetism",
        equation: "F = BIL\\sin\\theta",
        description: "Force on a straight wire carrying current in a magnetic field.",
        uses: ["force", "magnetic_field", "current", "length", "angle"],
        year: 1820, discoverer: "André-Marie Ampère"
    },
    {
        id: "inductor_energy",
        name: "Inductor Energy",
        field: "electromagnetism",
        equation: "U = \\tfrac{1}{2}LI^2",
        description: "Energy stored in the magnetic field of an inductor.",
        uses: ["energy", "inductance", "current"],
        year: 1886, discoverer: "Oliver Heaviside"
    },

    // ═════════════════════════════════════════════════════════════════════════
    // THERMODYNAMICS
    // ═════════════════════════════════════════════════════════════════════════

    {
        id: "ideal_gas_law",
        name: "Ideal Gas Law",
        field: "thermodynamics",
        equation: "PV = nRT",
        description: "Equation of state for an ideal gas, relating pressure, volume, temperature, and amount.",
        uses: ["pressure", "volume", "moles", "const_R", "temperature"],
        year: 1834, discoverer: "Benoît Paul Émile Clapeyron"
    },
    {
        id: "kinetic_theory_pressure",
        name: "Kinetic Theory of Pressure",
        field: "thermodynamics",
        equation: "P = \\frac{1}{3}\\rho\\langle v^2 \\rangle",
        description: "Pressure from molecular kinetic theory — connecting macroscopic pressure to molecular motion.",
        uses: ["pressure", "density", "velocity"],
        year: 1738, discoverer: "Daniel Bernoulli"
    },
    {
        id: "first_law_thermo",
        name: "First Law of Thermodynamics",
        field: "thermodynamics",
        equation: "\\Delta U = Q - W",
        description: "Energy conservation: change in internal energy equals heat added minus work done.",
        uses: ["internal_energy", "heat", "work"],
        year: 1850, discoverer: "Rudolf Clausius"
    },
    {
        id: "entropy_clausius",
        name: "Entropy (Clausius)",
        field: "thermodynamics",
        equation: "dS = \\frac{\\delta Q_{rev}}{T}",
        description: "Thermodynamic definition of entropy via reversible heat transfer.",
        uses: ["entropy", "heat", "temperature"],
        year: 1865, discoverer: "Rudolf Clausius"
    },
    {
        id: "boltzmann_entropy",
        name: "Boltzmann Entropy",
        field: "thermodynamics",
        equation: "S = k_B \\ln \\Omega",
        description: "Statistical definition of entropy — connecting thermodynamics to microscopic states. Engraved on Boltzmann's tombstone.",
        uses: ["entropy", "const_kB", "num_microstates"],
        year: 1877, discoverer: "Ludwig Boltzmann"
    },
    {
        id: "helmholtz_free_energy",
        name: "Helmholtz Free Energy",
        field: "thermodynamics",
        equation: "F = U - TS",
        description: "Maximum useful work obtainable at constant temperature and volume.",
        uses: ["helmholtz_energy", "internal_energy", "temperature", "entropy"],
        year: 1882, discoverer: "Hermann von Helmholtz"
    },
    {
        id: "gibbs_free_energy",
        name: "Gibbs Free Energy",
        field: "thermodynamics",
        equation: "G = H - TS",
        description: "Maximum non-expansion work at constant temperature and pressure. Determines spontaneity.",
        uses: ["gibbs_energy", "enthalpy", "temperature", "entropy"],
        year: 1876, discoverer: "Josiah Willard Gibbs"
    },
    {
        id: "enthalpy_def",
        name: "Enthalpy",
        field: "thermodynamics",
        equation: "H = U + PV",
        description: "Total heat content of a system at constant pressure.",
        uses: ["enthalpy", "internal_energy", "pressure", "volume"],
        year: 1850, discoverer: "Rudolf Clausius"
    },
    {
        id: "heat_capacity",
        name: "Heat Capacity",
        field: "thermodynamics",
        equation: "Q = mc_p\\Delta T",
        description: "Heat required to change the temperature of a substance.",
        uses: ["heat", "mass", "specific_heat", "temperature"],
        year: 1760, discoverer: "Joseph Black"
    },
    {
        id: "carnot_efficiency",
        name: "Carnot Efficiency",
        field: "thermodynamics",
        equation: "\\eta = 1 - \\frac{T_c}{T_h}",
        description: "Maximum possible efficiency of a heat engine operating between two temperatures.",
        uses: ["efficiency", "temperature"],
        year: 1824, discoverer: "Sadi Carnot"
    },
    {
        id: "stefan_boltzmann",
        name: "Stefan-Boltzmann Law",
        field: "thermodynamics",
        equation: "j = \\sigma T^4",
        description: "Total radiant power per unit area of a blackbody is proportional to T⁴.",
        uses: ["power", "area", "const_sigma", "temperature"],
        year: 1879, discoverer: "Josef Stefan & Ludwig Boltzmann"
    },
    {
        id: "wiens_law",
        name: "Wien's Displacement Law",
        field: "thermodynamics",
        equation: "\\lambda_{max} T = b",
        description: "Peak wavelength of blackbody radiation is inversely proportional to temperature.",
        uses: ["wavelength", "temperature", "const_bwien"],
        year: 1893, discoverer: "Wilhelm Wien"
    },
    {
        id: "planck_law",
        name: "Planck's Radiation Law",
        field: "thermodynamics",
        equation: "B(\\nu, T) = \\frac{2h\\nu^3}{c^2} \\cdot \\frac{1}{e^{h\\nu / k_B T} - 1}",
        description: "Spectral radiance of a blackbody — the equation that launched quantum mechanics.",
        uses: ["spectral_radiance", "const_h", "frequency", "const_c", "const_kB", "temperature"],
        year: 1900, discoverer: "Max Planck"
    },
    {
        id: "maxwell_boltzmann",
        name: "Maxwell-Boltzmann Distribution",
        field: "thermodynamics",
        equation: "f(v) = 4\\pi \\left(\\frac{m}{2\\pi k_B T}\\right)^{3/2} v^2 e^{-mv^2/(2k_B T)}",
        description: "Probability distribution of molecular speeds in an ideal gas.",
        uses: ["probability", "mass", "const_kB", "temperature", "velocity"],
        year: 1860, discoverer: "James Clerk Maxwell"
    },
    {
        id: "equipartition",
        name: "Equipartition Theorem",
        field: "thermodynamics",
        equation: "\\langle E \\rangle = \\frac{f}{2}k_B T",
        description: "Each quadratic degree of freedom contributes ½k_BT to average energy.",
        uses: ["energy", "degrees_of_freedom", "const_kB", "temperature"],
        year: 1845, discoverer: "John James Waterston"
    },
    {
        id: "fourier_law",
        name: "Fourier's Law of Heat Conduction",
        field: "thermodynamics",
        equation: "\\vec{q} = -\\kappa \\nabla T",
        description: "Heat flux is proportional to the negative temperature gradient.",
        uses: ["heat", "thermal_conductivity", "temperature"],
        year: 1822, discoverer: "Joseph Fourier"
    },
    {
        id: "clausius_clapeyron",
        name: "Clausius-Clapeyron Relation",
        field: "thermodynamics",
        equation: "\\frac{dP}{dT} = \\frac{\\Delta H_{vap}}{T\\Delta V}",
        description: "Relates vapor pressure change to temperature during phase transitions.",
        uses: ["pressure", "temperature", "enthalpy", "volume"],
        year: 1834, discoverer: "Benoît Paul Émile Clapeyron"
    },

    // ═════════════════════════════════════════════════════════════════════════
    // QUANTUM MECHANICS
    // ═════════════════════════════════════════════════════════════════════════

    {
        id: "planck_einstein",
        name: "Planck-Einstein Relation",
        field: "quantum_mechanics",
        equation: "E = h\\nu",
        description: "Energy of a photon is proportional to its frequency — the birth of quantum theory.",
        uses: ["energy", "const_h", "frequency"],
        year: 1905, discoverer: "Albert Einstein"
    },
    {
        id: "de_broglie",
        name: "de Broglie Wavelength",
        field: "quantum_mechanics",
        equation: "\\lambda = \\frac{h}{p}",
        description: "Every particle has a wavelength inversely proportional to its momentum — wave-particle duality.",
        uses: ["wavelength", "const_h", "momentum"],
        year: 1924, discoverer: "Louis de Broglie"
    },
    {
        id: "photon_momentum",
        name: "Photon Momentum",
        field: "quantum_mechanics",
        equation: "p = \\frac{h}{\\lambda} = \\frac{h\\nu}{c}",
        description: "Momentum carried by a massless photon.",
        uses: ["momentum", "const_h", "wavelength", "frequency", "const_c"],
        year: 1905, discoverer: "Albert Einstein"
    },
    {
        id: "heisenberg_xp",
        name: "Heisenberg Uncertainty (x-p)",
        field: "quantum_mechanics",
        equation: "\\Delta x \\, \\Delta p \\geq \\frac{\\hbar}{2}",
        description: "Position and momentum cannot both be known with arbitrary precision simultaneously.",
        uses: ["displacement", "momentum", "const_hbar"],
        year: 1927, discoverer: "Werner Heisenberg"
    },
    {
        id: "heisenberg_et",
        name: "Heisenberg Uncertainty (E-t)",
        field: "quantum_mechanics",
        equation: "\\Delta E \\, \\Delta t \\geq \\frac{\\hbar}{2}",
        description: "Energy and time cannot both be precisely determined — allows virtual particles.",
        uses: ["energy", "time", "const_hbar"],
        year: 1927, discoverer: "Werner Heisenberg"
    },
    {
        id: "schrodinger_td",
        name: "Schrödinger Equation (Time-Dependent)",
        field: "quantum_mechanics",
        equation: "i\\hbar \\frac{\\partial \\psi}{\\partial t} = \\hat{H}\\psi",
        description: "Fundamental equation of quantum mechanics describing the time evolution of a quantum state.",
        uses: ["const_hbar", "wave_function", "time", "energy"],
        year: 1926, discoverer: "Erwin Schrödinger"
    },
    {
        id: "schrodinger_ti",
        name: "Schrödinger Equation (Time-Independent)",
        field: "quantum_mechanics",
        equation: "\\hat{H}\\psi = E\\psi",
        description: "Eigenvalue equation for stationary quantum states.",
        uses: ["energy", "wave_function"],
        year: 1926, discoverer: "Erwin Schrödinger"
    },
    {
        id: "photoelectric",
        name: "Photoelectric Effect",
        field: "quantum_mechanics",
        equation: "E_k = h\\nu - \\phi",
        description: "Kinetic energy of emitted photoelectrons. Proved light comes in quanta (photons).",
        uses: ["energy", "const_h", "frequency", "work_function"],
        year: 1905, discoverer: "Albert Einstein"
    },
    {
        id: "compton_scattering",
        name: "Compton Scattering",
        field: "quantum_mechanics",
        equation: "\\Delta\\lambda = \\frac{h}{m_e c}(1 - \\cos\\theta)",
        description: "Wavelength shift when a photon scatters off an electron — proof of photon particle nature.",
        uses: ["wavelength", "const_h", "const_me", "const_c", "angle", "scattering_angle"],
        year: 1923, discoverer: "Arthur Compton"
    },
    {
        id: "hydrogen_energy",
        name: "Hydrogen Atom Energy Levels",
        field: "quantum_mechanics",
        equation: "E_n = -\\frac{m_e e^4}{8\\varepsilon_0^2 h^2} \\cdot \\frac{1}{n^2}",
        description: "Exact non-relativistic energy levels of the hydrogen atom.",
        uses: ["energy", "const_me", "const_e", "const_eps0", "const_h", "principal_quantum"],
        year: 1913, discoverer: "Niels Bohr"
    },
    {
        id: "bohr_radius",
        name: "Bohr Radius",
        field: "quantum_mechanics",
        equation: "a_0 = \\frac{4\\pi\\varepsilon_0 \\hbar^2}{m_e e^2}",
        description: "The most probable orbital radius in ground-state hydrogen — a natural length scale of QM.",
        uses: ["const_a0", "const_eps0", "const_hbar", "const_me", "const_e"],
        year: 1913, discoverer: "Niels Bohr"
    },
    {
        id: "fine_structure_constant",
        name: "Fine-Structure Constant",
        field: "quantum_mechanics",
        equation: "\\alpha = \\frac{e^2}{4\\pi\\varepsilon_0 \\hbar c}",
        description: "Dimensionless measure of electromagnetic coupling strength ≈ 1/137. One of the most fundamental numbers in nature.",
        uses: ["const_alpha", "const_e", "const_eps0", "const_hbar", "const_c"],
        year: 1916, discoverer: "Arnold Sommerfeld"
    },
    {
        id: "rydberg_formula",
        name: "Rydberg Formula",
        field: "quantum_mechanics",
        equation: "\\frac{1}{\\lambda} = R_\\infty \\left(\\frac{1}{n_1^2} - \\frac{1}{n_2^2}\\right)",
        description: "Predicts wavelengths of spectral lines for hydrogen-like atoms.",
        uses: ["wavelength", "const_Rinf", "principal_quantum"],
        year: 1888, discoverer: "Johannes Rydberg"
    },
    {
        id: "thermal_de_broglie",
        name: "Thermal de Broglie Wavelength",
        field: "quantum_mechanics",
        equation: "\\lambda_{th} = \\frac{h}{\\sqrt{2\\pi m k_B T}}",
        description: "Average quantum wavelength of particles at temperature T — bridges quantum and thermal physics.",
        uses: ["wavelength", "const_h", "mass", "const_kB", "temperature"],
        year: 1924, discoverer: "Louis de Broglie"
    },

    // ═════════════════════════════════════════════════════════════════════════
    // SPECIAL RELATIVITY
    // ═════════════════════════════════════════════════════════════════════════

    {
        id: "mass_energy",
        name: "Mass-Energy Equivalence",
        field: "special_relativity",
        equation: "E = mc^2",
        description: "Mass and energy are interchangeable — the most famous equation in physics.",
        uses: ["energy", "mass", "const_c"],
        year: 1905, discoverer: "Albert Einstein"
    },
    {
        id: "energy_momentum_relation",
        name: "Energy-Momentum Relation",
        field: "special_relativity",
        equation: "E^2 = (pc)^2 + (m_0 c^2)^2",
        description: "Full relativistic relationship between energy, momentum, and rest mass.",
        uses: ["energy", "momentum", "const_c", "mass"],
        year: 1905, discoverer: "Albert Einstein"
    },
    {
        id: "lorentz_factor_def",
        name: "Lorentz Factor",
        field: "special_relativity",
        equation: "\\gamma = \\frac{1}{\\sqrt{1 - v^2/c^2}}",
        description: "Factor by which time dilates and length contracts at relativistic speeds.",
        uses: ["lorentz_factor", "velocity", "const_c"],
        year: 1904, discoverer: "Hendrik Lorentz"
    },
    {
        id: "time_dilation",
        name: "Time Dilation",
        field: "special_relativity",
        equation: "\\Delta t' = \\gamma \\Delta t",
        description: "Moving clocks run slower — time itself is relative.",
        uses: ["time", "lorentz_factor"],
        year: 1905, discoverer: "Albert Einstein"
    },
    {
        id: "length_contraction",
        name: "Length Contraction",
        field: "special_relativity",
        equation: "L' = \\frac{L}{\\gamma}",
        description: "Moving objects are shortened along the direction of motion.",
        uses: ["length", "lorentz_factor"],
        year: 1905, discoverer: "Albert Einstein"
    },
    {
        id: "relativistic_momentum",
        name: "Relativistic Momentum",
        field: "special_relativity",
        equation: "\\vec{p} = \\gamma m\\vec{v}",
        description: "Momentum increases without bound as velocity approaches c.",
        uses: ["momentum", "lorentz_factor", "mass", "velocity"],
        year: 1905, discoverer: "Albert Einstein"
    },
    {
        id: "relativistic_ke",
        name: "Relativistic Kinetic Energy",
        field: "special_relativity",
        equation: "K = (\\gamma - 1)mc^2",
        description: "Kinetic energy in special relativity; reduces to ½mv² at low speeds.",
        uses: ["energy", "lorentz_factor", "mass", "const_c"],
        year: 1905, discoverer: "Albert Einstein"
    },
    {
        id: "relativistic_doppler",
        name: "Relativistic Doppler Effect",
        field: "special_relativity",
        equation: "f' = f\\sqrt{\\frac{1 + \\beta}{1 - \\beta}}",
        description: "Frequency shift for a source moving relative to an observer, accounting for time dilation.",
        uses: ["frequency", "velocity", "const_c"],
        year: 1905, discoverer: "Albert Einstein"
    },

    // ═════════════════════════════════════════════════════════════════════════
    // GENERAL RELATIVITY
    // ═════════════════════════════════════════════════════════════════════════

    {
        id: "einstein_field_equations",
        name: "Einstein Field Equations",
        field: "general_relativity",
        equation: "G_{\\mu\\nu} + \\Lambda g_{\\mu\\nu} = \\frac{8\\pi G}{c^4} T_{\\mu\\nu}",
        description: "Matter tells spacetime how to curve, spacetime tells matter how to move. The foundation of general relativity.",
        uses: ["einstein_tensor", "cosmological_constant", "metric_tensor", "const_G", "const_c", "stress_energy_tensor"],
        year: 1915, discoverer: "Albert Einstein"
    },
    {
        id: "schwarzschild_radius",
        name: "Schwarzschild Radius",
        field: "general_relativity",
        equation: "r_s = \\frac{2GM}{c^2}",
        description: "Radius at which escape velocity equals c — the event horizon of a black hole.",
        uses: ["distance", "const_G", "mass", "const_c"],
        year: 1916, discoverer: "Karl Schwarzschild"
    },
    {
        id: "gravitational_time_dilation",
        name: "Gravitational Time Dilation",
        field: "general_relativity",
        equation: "d\\tau = dt\\sqrt{1 - \\frac{2GM}{rc^2}}",
        description: "Clocks tick slower in stronger gravitational fields.",
        uses: ["proper_time", "time", "const_G", "mass", "distance", "const_c"],
        year: 1915, discoverer: "Albert Einstein"
    },
    {
        id: "gravitational_redshift",
        name: "Gravitational Redshift",
        field: "general_relativity",
        equation: "1 + z = \\frac{1}{\\sqrt{1 - r_s / r}}",
        description: "Light loses energy (redshifts) escaping a gravitational well.",
        uses: ["distance", "const_G", "mass", "const_c"],
        year: 1915, discoverer: "Albert Einstein"
    },
    {
        id: "geodesic_equation",
        name: "Geodesic Equation",
        field: "general_relativity",
        equation: "\\frac{d^2 x^\\mu}{d\\tau^2} + \\Gamma^\\mu_{\\alpha\\beta} \\frac{dx^\\alpha}{d\\tau}\\frac{dx^\\beta}{d\\tau} = 0",
        description: "Equation of motion for a free-falling particle in curved spacetime — gravity is geometry.",
        uses: ["distance", "time", "metric_tensor"],
        year: 1915, discoverer: "Albert Einstein"
    },

    // ═════════════════════════════════════════════════════════════════════════
    // CHEMISTRY
    // ═════════════════════════════════════════════════════════════════════════

    {
        id: "arrhenius",
        name: "Arrhenius Equation",
        field: "chemistry",
        equation: "k = A \\, e^{-E_a / RT}",
        description: "Temperature dependence of reaction rates — exponential sensitivity to activation energy.",
        uses: ["activation_energy", "const_R", "temperature"],
        year: 1889, discoverer: "Svante Arrhenius"
    },
    {
        id: "nernst",
        name: "Nernst Equation",
        field: "chemistry",
        equation: "E = E^\\circ - \\frac{RT}{nF}\\ln Q",
        description: "Cell potential under non-standard conditions — connects electrochemistry to thermodynamics.",
        uses: ["electric_potential", "const_R", "temperature", "moles", "const_F", "reaction_quotient"],
        year: 1889, discoverer: "Walther Nernst"
    },
    {
        id: "gibbs_reaction",
        name: "Gibbs Energy of Reaction",
        field: "chemistry",
        equation: "\\Delta G = \\Delta G^\\circ + RT\\ln Q",
        description: "Free energy change under non-standard conditions.",
        uses: ["gibbs_energy", "const_R", "temperature", "reaction_quotient"],
        year: 1876, discoverer: "Josiah Willard Gibbs"
    },
    {
        id: "equilibrium_gibbs",
        name: "Equilibrium & Gibbs Energy",
        field: "chemistry",
        equation: "\\Delta G^\\circ = -RT\\ln K",
        description: "At equilibrium, ΔG = 0. Links standard free energy to the equilibrium constant.",
        uses: ["gibbs_energy", "const_R", "temperature", "equilibrium_constant"],
        year: 1876, discoverer: "Josiah Willard Gibbs"
    },
    {
        id: "beer_lambert",
        name: "Beer-Lambert Law",
        field: "chemistry",
        equation: "A = \\varepsilon \\ell c",
        description: "Absorbance of light through a solution is proportional to concentration and path length.",
        uses: ["absorbance", "molar_absorptivity", "path_length", "concentration"],
        year: 1852, discoverer: "August Beer"
    },
    {
        id: "ideal_gas_molecular",
        name: "Ideal Gas Law (Molecular)",
        field: "chemistry",
        equation: "PV = Nk_BT",
        description: "Ideal gas law in terms of individual molecules rather than moles.",
        uses: ["pressure", "volume", "num_particles", "const_kB", "temperature"],
        year: 1834, discoverer: "Benoît Paul Émile Clapeyron"
    },
    {
        id: "faraday_electrolysis",
        name: "Faraday's Law of Electrolysis",
        field: "chemistry",
        equation: "m = \\frac{MIt}{nF}",
        description: "Mass deposited in electrolysis depends on current, time, and Faraday constant.",
        uses: ["mass", "current", "time", "moles", "const_F"],
        year: 1834, discoverer: "Michael Faraday"
    },
    {
        id: "raoults_law",
        name: "Raoult's Law",
        field: "chemistry",
        equation: "P_A = x_A P_A^\\circ",
        description: "Partial vapor pressure of a component in an ideal solution.",
        uses: ["pressure", "concentration"],
        year: 1887, discoverer: "François-Marie Raoult"
    },

    // ═════════════════════════════════════════════════════════════════════════
    // STATISTICAL MECHANICS
    // ═════════════════════════════════════════════════════════════════════════

    {
        id: "partition_function",
        name: "Canonical Partition Function",
        field: "statistical_mechanics",
        equation: "Z = \\sum_i e^{-E_i / k_B T}",
        description: "Sum over all quantum states — encodes all thermodynamic information of a system.",
        uses: ["var_partition_func", "energy", "const_kB", "temperature"],
        year: 1902, discoverer: "Josiah Willard Gibbs"
    },
    {
        id: "boltzmann_distribution",
        name: "Boltzmann Distribution",
        field: "statistical_mechanics",
        equation: "P_i = \\frac{e^{-E_i / k_B T}}{Z}",
        description: "Probability of a system being in a microstate with energy E_i at thermal equilibrium.",
        uses: ["probability", "energy", "const_kB", "temperature", "var_partition_func"],
        year: 1868, discoverer: "Ludwig Boltzmann"
    },
    {
        id: "free_energy_partition",
        name: "Free Energy from Partition Function",
        field: "statistical_mechanics",
        equation: "F = -k_B T \\ln Z",
        description: "Direct bridge from statistical mechanics to thermodynamics.",
        uses: ["helmholtz_energy", "const_kB", "temperature", "var_partition_func"],
        year: 1902, discoverer: "Josiah Willard Gibbs"
    },
    {
        id: "fermi_dirac",
        name: "Fermi-Dirac Distribution",
        field: "statistical_mechanics",
        equation: "f(E) = \\frac{1}{e^{(E - \\mu)/k_B T} + 1}",
        description: "Probability of a fermion state being occupied — governs electrons in metals, white dwarfs, neutron stars.",
        uses: ["probability", "energy", "chemical_potential", "const_kB", "temperature"],
        year: 1926, discoverer: "Enrico Fermi & Paul Dirac"
    },
    {
        id: "bose_einstein",
        name: "Bose-Einstein Distribution",
        field: "statistical_mechanics",
        equation: "\\langle n \\rangle = \\frac{1}{e^{(E - \\mu)/k_B T} - 1}",
        description: "Average occupation number for bosons — governs photons, phonons, and Bose-Einstein condensates.",
        uses: ["num_particles", "energy", "chemical_potential", "const_kB", "temperature"],
        year: 1924, discoverer: "Satyendra Nath Bose & Albert Einstein"
    },
    {
        id: "stefan_boltzmann_derivation",
        name: "Stefan-Boltzmann from Constants",
        field: "statistical_mechanics",
        equation: "\\sigma = \\frac{2\\pi^5 k_B^4}{15 h^3 c^2}",
        description: "The Stefan-Boltzmann constant derived from fundamental constants — a hidden bridge connecting thermal radiation, quantum mechanics, and electromagnetism.",
        uses: ["const_sigma", "const_kB", "const_h", "const_c"],
        year: 1884, discoverer: "Ludwig Boltzmann"
    },
    {
        id: "sackur_tetrode",
        name: "Sackur-Tetrode Equation",
        field: "statistical_mechanics",
        equation: "S = Nk_B\\left[\\frac{5}{2} + \\ln\\left(\\frac{V}{N}\\left(\\frac{4\\pi m E}{3Nh^2}\\right)^{3/2}\\right)\\right]",
        description: "Exact entropy of an ideal monatomic gas from first principles of quantum statistical mechanics.",
        uses: ["entropy", "num_particles", "const_kB", "volume", "mass", "energy", "const_h"],
        year: 1912, discoverer: "Hugo Tetrode & Otto Sackur"
    },

    // ═════════════════════════════════════════════════════════════════════════
    // FLUID MECHANICS
    // ═════════════════════════════════════════════════════════════════════════

    {
        id: "bernoulli",
        name: "Bernoulli's Equation",
        field: "fluid_mechanics",
        equation: "P + \\tfrac{1}{2}\\rho v^2 + \\rho g h = \\text{const}",
        description: "Conservation of energy in fluid flow — connects pressure, velocity, and elevation.",
        uses: ["pressure", "density", "velocity", "const_g", "height"],
        year: 1738, discoverer: "Daniel Bernoulli"
    },
    {
        id: "continuity",
        name: "Continuity Equation (Fluids)",
        field: "fluid_mechanics",
        equation: "A_1 v_1 = A_2 v_2",
        description: "Conservation of mass in incompressible fluid flow.",
        uses: ["area", "velocity"],
        year: 1738, discoverer: "Daniel Bernoulli"
    },
    {
        id: "reynolds_number",
        name: "Reynolds Number",
        field: "fluid_mechanics",
        equation: "Re = \\frac{\\rho v L}{\\mu}",
        description: "Ratio of inertial to viscous forces — predicts laminar vs. turbulent flow.",
        uses: ["var_reynolds", "density", "velocity", "length", "viscosity"],
        year: 1883, discoverer: "Osborne Reynolds"
    },
    {
        id: "stokes_law",
        name: "Stokes' Law",
        field: "fluid_mechanics",
        equation: "F_d = 6\\pi \\mu r v",
        description: "Viscous drag force on a sphere at low Reynolds number.",
        uses: ["force", "viscosity", "distance", "velocity"],
        year: 1851, discoverer: "George Gabriel Stokes"
    },
    {
        id: "archimedes",
        name: "Archimedes' Principle",
        field: "fluid_mechanics",
        equation: "F_b = \\rho_f V g",
        description: "Buoyant force equals the weight of displaced fluid.",
        uses: ["force", "density", "volume", "const_g"],
        year: -250, discoverer: "Archimedes"
    },
    {
        id: "poiseuille",
        name: "Poiseuille's Law",
        field: "fluid_mechanics",
        equation: "Q = \\frac{\\pi r^4 \\Delta P}{8 \\mu L}",
        description: "Volumetric flow rate through a cylindrical pipe — fourth power dependence on radius.",
        uses: ["flow_rate", "distance", "pressure", "viscosity", "length"],
        year: 1838, discoverer: "Jean Léonard Marie Poiseuille"
    },

    // ═════════════════════════════════════════════════════════════════════════
    // OPTICS & WAVES
    // ═════════════════════════════════════════════════════════════════════════

    {
        id: "wave_equation",
        name: "Wave Relation",
        field: "optics_waves",
        equation: "v = \\nu \\lambda",
        description: "Fundamental relationship between wave speed, frequency, and wavelength.",
        uses: ["velocity", "frequency", "wavelength"],
        year: 1678, discoverer: "Christiaan Huygens"
    },
    {
        id: "snells_law",
        name: "Snell's Law",
        field: "optics_waves",
        equation: "n_1 \\sin\\theta_1 = n_2 \\sin\\theta_2",
        description: "Law of refraction relating angles and refractive indices at an interface.",
        uses: ["refractive_index", "angle"],
        year: 1621, discoverer: "Willebrord Snellius"
    },
    {
        id: "diffraction_grating",
        name: "Diffraction Grating Equation",
        field: "optics_waves",
        equation: "d\\sin\\theta = m\\lambda",
        description: "Condition for constructive interference from a diffraction grating.",
        uses: ["distance", "angle", "wavelength"],
        year: 1821, discoverer: "Joseph von Fraunhofer"
    },
    {
        id: "doppler_classical",
        name: "Doppler Effect (Classical)",
        field: "optics_waves",
        equation: "f' = f \\frac{v \\pm v_o}{v \\mp v_s}",
        description: "Frequency shift due to relative motion of source and observer.",
        uses: ["frequency", "velocity"],
        year: 1842, discoverer: "Christian Doppler"
    },
    {
        id: "thin_lens",
        name: "Thin Lens Equation",
        field: "optics_waves",
        equation: "\\frac{1}{f} = \\frac{1}{d_o} + \\frac{1}{d_i}",
        description: "Relates focal length to object and image distances.",
        uses: ["distance"],
        year: 1693, discoverer: "Edmond Halley"
    },
    {
        id: "malus_law",
        name: "Malus's Law",
        field: "optics_waves",
        equation: "I = I_0 \\cos^2\\theta",
        description: "Intensity of polarized light after passing through a polarizer.",
        uses: ["intensity", "angle"],
        year: 1809, discoverer: "Étienne-Louis Malus"
    },
    {
        id: "rayleigh_scattering",
        name: "Rayleigh Scattering",
        field: "optics_waves",
        equation: "I \\propto \\frac{1}{\\lambda^4}",
        description: "Scattering intensity inversely proportional to λ⁴ — why the sky is blue.",
        uses: ["intensity", "wavelength"],
        year: 1871, discoverer: "Lord Rayleigh"
    },

    // ═════════════════════════════════════════════════════════════════════════
    // NUCLEAR & PARTICLE PHYSICS
    // ═════════════════════════════════════════════════════════════════════════

    {
        id: "radioactive_decay",
        name: "Radioactive Decay Law",
        field: "nuclear_physics",
        equation: "N(t) = N_0 \\, e^{-\\lambda t}",
        description: "Number of undecayed nuclei decreases exponentially over time.",
        uses: ["num_particles", "decay_constant", "time"],
        year: 1903, discoverer: "Ernest Rutherford & Frederick Soddy"
    },
    {
        id: "half_life_relation",
        name: "Half-Life",
        field: "nuclear_physics",
        equation: "t_{1/2} = \\frac{\\ln 2}{\\lambda}",
        description: "Time for half of a radioactive sample to decay.",
        uses: ["half_life", "decay_constant"],
        year: 1903, discoverer: "Ernest Rutherford"
    },
    {
        id: "binding_energy",
        name: "Nuclear Binding Energy",
        field: "nuclear_physics",
        equation: "E_b = \\Delta m \\cdot c^2",
        description: "Energy equivalent of the mass defect — the glue that holds nuclei together.",
        uses: ["energy", "mass", "const_c"],
        year: 1905, discoverer: "Albert Einstein"
    },
    {
        id: "nuclear_radius",
        name: "Nuclear Radius",
        field: "nuclear_physics",
        equation: "r = r_0 A^{1/3}",
        description: "Nuclear radius scales with the cube root of mass number, confirming nuclear density is constant.",
        uses: ["distance", "num_particles"],
        year: 1911, discoverer: "Ernest Rutherford"
    },
    {
        id: "decay_activity",
        name: "Radioactive Activity",
        field: "nuclear_physics",
        equation: "A = \\lambda N",
        description: "Activity (decays per second) equals decay constant times number of nuclei.",
        uses: ["decay_constant", "num_particles"],
        year: 1903, discoverer: "Ernest Rutherford"
    },

    // ═════════════════════════════════════════════════════════════════════════
    // ASTROPHYSICS & COSMOLOGY
    // ═════════════════════════════════════════════════════════════════════════

    {
        id: "hubbles_law",
        name: "Hubble's Law",
        field: "astrophysics",
        equation: "v = H_0 d",
        description: "Recession velocity of galaxies is proportional to their distance — the expanding universe.",
        uses: ["velocity", "hubble_constant", "distance"],
        year: 1929, discoverer: "Edwin Hubble"
    },
    {
        id: "stellar_luminosity",
        name: "Stellar Luminosity",
        field: "astrophysics",
        equation: "L = 4\\pi R^2 \\sigma T^4",
        description: "Total power output of a star treated as a blackbody.",
        uses: ["luminosity", "radius", "const_sigma", "temperature"],
        year: 1879, discoverer: "Josef Stefan"
    },
    {
        id: "friedmann",
        name: "Friedmann Equation",
        field: "astrophysics",
        equation: "H^2 = \\frac{8\\pi G}{3}\\rho - \\frac{kc^2}{a^2} + \\frac{\\Lambda c^2}{3}",
        description: "Governs the expansion rate of the universe — foundation of Big Bang cosmology.",
        uses: ["hubble_constant", "const_G", "density", "const_c", "scale_factor", "cosmological_constant"],
        year: 1922, discoverer: "Alexander Friedmann"
    },
    {
        id: "hawking_temperature",
        name: "Hawking Temperature",
        field: "astrophysics",
        equation: "T_H = \\frac{\\hbar c^3}{8\\pi G M k_B}",
        description: "Black holes have a temperature — connecting gravity, quantum mechanics, and thermodynamics in one equation.",
        uses: ["temperature", "const_hbar", "const_c", "const_G", "mass", "const_kB"],
        year: 1974, discoverer: "Stephen Hawking"
    },
    {
        id: "gravitational_wave_power",
        name: "Gravitational Wave Luminosity",
        field: "astrophysics",
        equation: "P = -\\frac{32 G^4}{5 c^5}\\frac{(m_1 m_2)^2(m_1 + m_2)}{r^5}",
        description: "Power radiated as gravitational waves by a binary system.",
        uses: ["power", "const_G", "const_c", "mass", "distance"],
        year: 1918, discoverer: "Albert Einstein"
    },
    {
        id: "chandrasekhar_mass",
        name: "Chandrasekhar Limit",
        field: "astrophysics",
        equation: "M_{Ch} = \\frac{\\omega_3^0 \\sqrt{3\\pi}}{2}\\left(\\frac{\\hbar c}{G}\\right)^{3/2}\\frac{1}{(\\mu_e m_p)^2}",
        description: "Maximum mass of a stable white dwarf star ≈ 1.4 M☉ — beyond this, gravitational collapse wins.",
        uses: ["mass", "const_hbar", "const_c", "const_G", "const_mp"],
        year: 1931, discoverer: "Subrahmanyan Chandrasekhar"
    },

    // ═════════════════════════════════════════════════════════════════════════
    // INFORMATION THEORY
    // ═════════════════════════════════════════════════════════════════════════

    {
        id: "shannon_entropy",
        name: "Shannon Entropy",
        field: "information_theory",
        equation: "H = -\\sum_i p_i \\log_2 p_i",
        description: "Measure of information content or uncertainty. Foundation of information theory.",
        uses: ["information_entropy", "probability"],
        year: 1948, discoverer: "Claude Shannon"
    },
    {
        id: "landauer_principle",
        name: "Landauer's Principle",
        field: "information_theory",
        equation: "E_{min} = k_B T \\ln 2",
        description: "Minimum energy dissipated when erasing one bit — connects information theory to thermodynamics.",
        uses: ["energy", "const_kB", "temperature"],
        year: 1961, discoverer: "Rolf Landauer"
    },
    {
        id: "bekenstein_hawking",
        name: "Bekenstein-Hawking Entropy",
        field: "information_theory",
        equation: "S_{BH} = \\frac{k_B A c^3}{4 G \\hbar}",
        description: "Entropy of a black hole is proportional to its horizon area — uniting gravity, quantum mechanics, thermodynamics, and information.",
        uses: ["entropy", "const_kB", "area", "const_c", "const_G", "const_hbar"],
        year: 1973, discoverer: "Jacob Bekenstein & Stephen Hawking"
    },

    // ═════════════════════════════════════════════════════════════════════════
    // META: DERIVED CONSTANT RELATIONSHIPS
    // These reveal hidden bridges between fundamental constants
    // ═════════════════════════════════════════════════════════════════════════

    {
        id: "gas_constant_relation",
        name: "Gas Constant = N_A × k_B",
        field: "meta",
        equation: "R = N_A k_B",
        description: "The gas constant is Avogadro's number times Boltzmann's constant — bridging molar and molecular scales.",
        uses: ["const_R", "const_NA", "const_kB"],
        year: 1900, discoverer: "Max Planck"
    },
    {
        id: "faraday_constant_relation",
        name: "Faraday Constant = N_A × e",
        field: "meta",
        equation: "F = N_A e",
        description: "The Faraday constant is one mole of elementary charges — bridging electrochemistry and particle physics.",
        uses: ["const_F", "const_NA", "const_e"],
        year: 1834, discoverer: "Michael Faraday"
    },
    {
        id: "coulomb_constant_relation",
        name: "Coulomb Constant from ε₀",
        field: "meta",
        equation: "k_e = \\frac{1}{4\\pi\\varepsilon_0}",
        description: "Coulomb's constant expressed in terms of vacuum permittivity.",
        uses: ["const_ke", "const_eps0"],
        year: 1785, discoverer: "Charles-Augustin de Coulomb"
    },
    {
        id: "reduced_planck_relation",
        name: "Reduced Planck Constant",
        field: "meta",
        equation: "\\hbar = \\frac{h}{2\\pi}",
        description: "The reduced Planck constant — natural unit for angular quantities in quantum mechanics.",
        uses: ["const_hbar", "const_h"],
        year: 1900, discoverer: "Max Planck"
    },
    {
        id: "rydberg_from_constants",
        name: "Rydberg from Fundamental Constants",
        field: "meta",
        equation: "R_\\infty = \\frac{m_e e^4}{8\\varepsilon_0^2 h^3 c}",
        description: "The Rydberg constant expressed in terms of electron mass, charge, permittivity, Planck's constant, and speed of light.",
        uses: ["const_Rinf", "const_me", "const_e", "const_eps0", "const_h", "const_c"],
        year: 1913, discoverer: "Niels Bohr"
    },
    {
        id: "wien_from_constants",
        name: "Wien Constant from Fundamentals",
        field: "meta",
        equation: "b \\approx \\frac{hc}{4.965 \\, k_B}",
        description: "Wien's displacement constant derived from Planck's constant, speed of light, and Boltzmann constant.",
        uses: ["const_bwien", "const_h", "const_c", "const_kB"],
        year: 1893, discoverer: "Wilhelm Wien"
    },
    {
        id: "bohr_magneton",
        name: "Bohr Magneton",
        field: "meta",
        equation: "\\mu_B = \\frac{e\\hbar}{2m_e}",
        description: "Natural unit of magnetic moment for electrons — connects electromagnetism and quantum mechanics.",
        uses: ["const_e", "const_hbar", "const_me"],
        year: 1913, discoverer: "Niels Bohr"
    },
    {
        id: "planck_units",
        name: "Planck Length",
        field: "meta",
        equation: "\\ell_P = \\sqrt{\\frac{\\hbar G}{c^3}}",
        description: "The smallest meaningful length scale in physics — where quantum mechanics and gravity meet.",
        uses: ["const_hbar", "const_G", "const_c"],
        year: 1899, discoverer: "Max Planck"
    },
    {
        id: "planck_temperature",
        name: "Planck Temperature",
        field: "meta",
        equation: "T_P = \\sqrt{\\frac{\\hbar c^5}{G k_B^2}}",
        description: "The highest physically meaningful temperature — connecting all four fundamental constants.",
        uses: ["temperature", "const_hbar", "const_c", "const_G", "const_kB"],
        year: 1899, discoverer: "Max Planck"
    },
    {
        id: "planck_mass",
        name: "Planck Mass",
        field: "meta",
        equation: "m_P = \\sqrt{\\frac{\\hbar c}{G}}",
        description: "Natural unit of mass at the quantum-gravitational scale.",
        uses: ["mass", "const_hbar", "const_c", "const_G"],
        year: 1899, discoverer: "Max Planck"
    },

    // ═════════════════════════════════════════════════════════════════════════
    // ADDITIONAL EQUATIONS — filling low-connectivity nodes
    // ═════════════════════════════════════════════════════════════════════════

    // --- Connects: acceleration, force, mass (acceleration was 1-conn) ---
    {
        id: "weight",
        name: "Weight",
        field: "classical_mechanics",
        equation: "W = mg",
        description: "The gravitational force on a mass near Earth's surface.",
        uses: ["force", "mass", "const_g", "acceleration"],
        year: 1687, discoverer: "Isaac Newton"
    },
    // --- Connects: angular_momentum, momentum, distance ---
    {
        id: "angular_momentum_particle",
        name: "Angular Momentum (Particle)",
        field: "classical_mechanics",
        equation: "L = mvr",
        description: "Angular momentum of a particle moving in a circle of radius r.",
        uses: ["angular_momentum", "mass", "velocity", "distance"],
        year: 1687, discoverer: "Isaac Newton"
    },
    // --- Connects: torque, force, distance, angle ---
    {
        id: "torque_cross",
        name: "Torque (Cross Product)",
        field: "classical_mechanics",
        equation: "\\tau = rF\\sin\\theta",
        description: "Torque as the cross product of lever arm and applied force.",
        uses: ["torque", "distance", "force", "angle"],
        year: 1687, discoverer: "Isaac Newton"
    },
    // --- Connects: angular_acceleration, angular_velocity, time ---
    {
        id: "angular_kinematics",
        name: "Angular Kinematics",
        field: "classical_mechanics",
        equation: "\\omega = \\omega_0 + \\alpha t",
        description: "Angular velocity under constant angular acceleration.",
        uses: ["angular_velocity", "angular_acceleration", "time"],
        year: 1736, discoverer: "Leonhard Euler"
    },
    // --- Connects: radius, luminosity (luminosity was 1-conn, radius was 1-conn) ---
    {
        id: "inverse_square_intensity",
        name: "Inverse Square Law (Light)",
        field: "astrophysics",
        equation: "I = \\frac{L}{4\\pi r^2}",
        description: "Intensity of radiation decreases with the square of the distance from the source.",
        uses: ["intensity", "luminosity", "distance"],
        year: 1604, discoverer: "Johannes Kepler"
    },
    // --- Connects: efficiency, power, heat ---
    {
        id: "heat_engine_efficiency",
        name: "Heat Engine Efficiency",
        field: "thermodynamics",
        equation: "\\eta = \\frac{W}{Q_H}",
        description: "Thermal efficiency is the ratio of work output to heat input.",
        uses: ["efficiency", "work", "heat"],
        year: 1824, discoverer: "Sadi Carnot"
    },
    // --- Connects: resistance, electric_field, current ---
    {
        id: "resistivity",
        name: "Resistance from Geometry",
        field: "electromagnetism",
        equation: "R = \\frac{\\rho L}{A}",
        description: "Resistance depends on material resistivity, length, and cross-sectional area.",
        uses: ["resistance", "density", "length", "area"],
        year: 1827, discoverer: "Georg Simon Ohm"
    },
    // --- Connects: joule_heating power, current, resistance ---
    {
        id: "joule_heating",
        name: "Joule Heating",
        field: "electromagnetism",
        equation: "P = I^2 R",
        description: "Power dissipated as heat in a resistor.",
        uses: ["power", "current", "resistance"],
        year: 1841, discoverer: "James Prescott Joule"
    },
    // --- Connects: inductance, emf, current, time ---
    {
        id: "inductor_emf",
        name: "Inductor EMF",
        field: "electromagnetism",
        equation: "\\mathcal{E} = -L\\frac{dI}{dt}",
        description: "Self-induced EMF in an inductor opposes changes in current.",
        uses: ["emf", "inductance", "current", "time"],
        year: 1831, discoverer: "Michael Faraday"
    },
    // --- Connects: magnetic_flux, magnetic_field, area, angle ---
    {
        id: "magnetic_flux_def",
        name: "Magnetic Flux",
        field: "electromagnetism",
        equation: "\\Phi_B = BA\\cos\\theta",
        description: "Magnetic flux through a surface — the total magnetic field passing through an area.",
        uses: ["magnetic_flux", "magnetic_field", "area", "angle"],
        year: 1831, discoverer: "Michael Faraday"
    },
    // --- Connects: refractive_index, velocity, speed of light ---
    {
        id: "refractive_index_def",
        name: "Refractive Index",
        field: "optics_waves",
        equation: "n = \\frac{c}{v}",
        description: "The refractive index is the ratio of the speed of light in vacuum to its speed in a medium.",
        uses: ["refractive_index", "const_c", "velocity"],
        year: 1621, discoverer: "Willebrord Snellius"
    },
    // --- Connects: spectral_radiance (was 1-conn), connects to more vars ---
    {
        id: "rayleigh_jeans",
        name: "Rayleigh-Jeans Law",
        field: "optics_waves",
        equation: "B(\\nu, T) = \\frac{2\\nu^2 k_B T}{c^2}",
        description: "Classical limit of Planck's law at low frequencies — its failure at high frequencies led to quantum theory (ultraviolet catastrophe).",
        uses: ["spectral_radiance", "frequency", "const_kB", "temperature", "const_c"],
        year: 1900, discoverer: "Lord Rayleigh & James Jeans"
    },
    // --- Connects: specific_heat, internal_energy, temperature ---
    {
        id: "dulong_petit",
        name: "Dulong-Petit Law",
        field: "thermodynamics",
        equation: "c_v = \\frac{3R}{M}",
        description: "Molar heat capacity of crystalline solids approaches 3R at high temperatures.",
        uses: ["specific_heat", "const_R"],
        year: 1819, discoverer: "Pierre Louis Dulong & Alexis Thérèse Petit"
    },
    // --- Connects: thermal_conductivity, heat, area, temperature, length ---
    {
        id: "fourier_steady",
        name: "Fourier's Law (Steady State)",
        field: "thermodynamics",
        equation: "Q = \\frac{\\kappa A \\Delta T \\, t}{L}",
        description: "Steady-state heat conduction through a slab of material.",
        uses: ["heat", "thermal_conductivity", "area", "temperature", "time", "length"],
        year: 1822, discoverer: "Joseph Fourier"
    },
    // --- Connects: num_microstates, num_particles ---
    {
        id: "ideal_gas_microstates",
        name: "Ideal Gas Multiplicity",
        field: "statistical_mechanics",
        equation: "\\Omega \\propto V^N E^{3N/2}",
        description: "Number of microstates for an ideal gas — proportional to volume and energy raised to powers of N.",
        uses: ["num_microstates", "volume", "num_particles", "energy"],
        year: 1877, discoverer: "Ludwig Boltzmann"
    },
    // --- Connects: degrees_of_freedom, internal_energy, temperature ---
    {
        id: "internal_energy_ideal",
        name: "Internal Energy (Ideal Gas)",
        field: "thermodynamics",
        equation: "U = \\frac{f}{2}nRT",
        description: "Internal energy of an ideal gas depends on degrees of freedom and temperature.",
        uses: ["internal_energy", "degrees_of_freedom", "moles", "const_R", "temperature"],
        year: 1845, discoverer: "John James Waterston"
    },
    // --- Connects: reynolds_number, flow_rate ---
    {
        id: "drag_force",
        name: "Drag Force (High Re)",
        field: "fluid_mechanics",
        equation: "F_d = \\tfrac{1}{2} C_d \\rho A v^2",
        description: "Aerodynamic drag force at high Reynolds numbers — proportional to velocity squared.",
        uses: ["force", "density", "area", "velocity"],
        year: 1883, discoverer: "Lord Rayleigh"
    },
    // --- Connects: flow_rate, area, velocity ---
    {
        id: "flow_rate_def",
        name: "Volumetric Flow Rate",
        field: "fluid_mechanics",
        equation: "Q = Av",
        description: "Flow rate equals cross-sectional area times fluid velocity.",
        uses: ["flow_rate", "area", "velocity"],
        year: 1738, discoverer: "Daniel Bernoulli"
    },
    // --- Connects: half_life, decay_constant, time ---
    {
        id: "decay_by_half_life",
        name: "Decay by Half-Life",
        field: "nuclear_physics",
        equation: "N(t) = N_0 \\left(\\frac{1}{2}\\right)^{t/t_{1/2}}",
        description: "Radioactive decay expressed in terms of half-life.",
        uses: ["num_particles", "time", "half_life"],
        year: 1903, discoverer: "Ernest Rutherford"
    },
    // --- Connects: work_function, electric_potential, charge ---
    {
        id: "threshold_frequency",
        name: "Threshold Frequency",
        field: "quantum_mechanics",
        equation: "\\phi = h\\nu_0",
        description: "Work function equals Planck's constant times the minimum frequency for photoemission.",
        uses: ["work_function", "const_h", "frequency"],
        year: 1905, discoverer: "Albert Einstein"
    },
    // --- Connects: activation_energy, temperature ---
    {
        id: "eyring",
        name: "Eyring Equation",
        field: "chemistry",
        equation: "k = \\frac{k_B T}{h} e^{-\\Delta G^\\ddagger / RT}",
        description: "Rate constant from transition state theory — connecting thermodynamics, quantum mechanics, and kinetics.",
        uses: ["const_kB", "temperature", "const_h", "activation_energy", "const_R"],
        year: 1935, discoverer: "Henry Eyring"
    },
    // --- Connects: equilibrium_constant, concentration ---
    {
        id: "equilibrium_expression",
        name: "Equilibrium Expression",
        field: "chemistry",
        equation: "K = \\frac{[C]^c[D]^d}{[A]^a[B]^b}",
        description: "The equilibrium constant expressed as a ratio of product to reactant concentrations.",
        uses: ["equilibrium_constant", "concentration"],
        year: 1864, discoverer: "Cato Guldberg & Peter Waage"
    },
    // --- Connects: information_entropy, num_microstates ---
    {
        id: "entropy_information_bridge",
        name: "Gibbs Entropy",
        field: "statistical_mechanics",
        equation: "S = -k_B \\sum_i p_i \\ln p_i",
        description: "General statistical entropy — identical in form to Shannon entropy, revealing the deep link between physics and information.",
        uses: ["entropy", "const_kB", "probability", "information_entropy"],
        year: 1902, discoverer: "Josiah Willard Gibbs"
    },
    // --- Connects: const_alpha, const_a0, const_Rinf (all low-conn constants) ---
    {
        id: "rydberg_fine_structure",
        name: "Rydberg from Fine Structure",
        field: "meta",
        equation: "R_\\infty = \\frac{\\alpha^2 m_e c}{2h}",
        description: "Rydberg constant expressed via the fine-structure constant — linking atomic spectra to fundamental QED coupling.",
        uses: ["const_Rinf", "const_alpha", "const_me", "const_c", "const_h"],
        year: 1916, discoverer: "Arnold Sommerfeld"
    },
    {
        id: "bohr_radius_fine_structure",
        name: "Bohr Radius from Fine Structure",
        field: "meta",
        equation: "a_0 = \\frac{\\hbar}{\\alpha m_e c}",
        description: "Bohr radius expressed via the fine-structure constant.",
        uses: ["const_a0", "const_hbar", "const_alpha", "const_me", "const_c"],
        year: 1913, discoverer: "Niels Bohr"
    },
    // --- Connects: const_mp more ---
    {
        id: "proton_compton",
        name: "Proton Compton Wavelength",
        field: "quantum_mechanics",
        equation: "\\lambda_p = \\frac{h}{m_p c}",
        description: "Compton wavelength of the proton — the quantum length scale of the proton.",
        uses: ["wavelength", "const_h", "const_mp", "const_c"],
        year: 1923, discoverer: "Arthur Compton"
    },
    // --- Connects: scale_factor, hubble_constant ---
    {
        id: "hubble_scale",
        name: "Hubble Parameter (General)",
        field: "astrophysics",
        equation: "H(t) = \\frac{\\dot{a}}{a}",
        description: "The Hubble parameter as the fractional rate of change of the scale factor.",
        uses: ["hubble_constant", "scale_factor", "time"],
        year: 1922, discoverer: "Alexander Friedmann"
    },
    // --- Connects: stress_energy_tensor more ---
    {
        id: "perfect_fluid",
        name: "Perfect Fluid Stress-Energy",
        field: "general_relativity",
        equation: "T^{\\mu\\nu} = (\\rho + P/c^2)u^\\mu u^\\nu + P g^{\\mu\\nu}",
        description: "Stress-energy tensor for a perfect fluid — the source term in Einstein's equations for cosmology.",
        uses: ["stress_energy_tensor", "density", "pressure", "const_c", "metric_tensor", "velocity"],
        year: 1915, discoverer: "Albert Einstein"
    },
    // --- Connects: einstein_tensor, metric_tensor ---
    {
        id: "einstein_tensor_def",
        name: "Einstein Tensor",
        field: "general_relativity",
        equation: "G_{\\mu\\nu} = R_{\\mu\\nu} - \\tfrac{1}{2}R \\, g_{\\mu\\nu}",
        description: "The Einstein tensor encodes spacetime curvature in a divergence-free form.",
        uses: ["einstein_tensor", "metric_tensor"],
        year: 1915, discoverer: "Albert Einstein"
    },
    // --- Connects: chemical_potential, gibbs_energy, moles ---
    {
        id: "chemical_potential_def",
        name: "Chemical Potential",
        field: "thermodynamics",
        equation: "\\mu = \\left(\\frac{\\partial G}{\\partial N}\\right)_{T,P}",
        description: "Chemical potential is the change in Gibbs energy per particle added — drives diffusion and reactions.",
        uses: ["chemical_potential", "gibbs_energy", "num_particles", "temperature", "pressure"],
        year: 1876, discoverer: "Josiah Willard Gibbs"
    },
    // --- Connects: cosmological_constant more ---
    {
        id: "dark_energy_density",
        name: "Dark Energy Density",
        field: "astrophysics",
        equation: "\\rho_\\Lambda = \\frac{\\Lambda c^2}{8\\pi G}",
        description: "Energy density of the cosmological constant — the vacuum energy driving accelerated expansion.",
        uses: ["density", "cosmological_constant", "const_c", "const_G"],
        year: 1998, discoverer: "Saul Perlmutter, Brian Schmidt & Adam Riess"
    },
    // --- Connects: absorbance, intensity ---
    {
        id: "absorbance_def",
        name: "Absorbance Definition",
        field: "chemistry",
        equation: "A = -\\log_{10}\\frac{I}{I_0}",
        description: "Absorbance as the negative log of transmittance — connects spectroscopy to light intensity.",
        uses: ["absorbance", "intensity"],
        year: 1729, discoverer: "Pierre Bouguer"
    },
    // --- Connects: radius, volume ---
    {
        id: "sphere_volume",
        name: "Volume of a Sphere",
        field: "classical_mechanics",
        equation: "V = \\frac{4}{3}\\pi R^3",
        description: "Volume of a sphere — fundamental geometric relation used across physics.",
        uses: ["volume", "radius"],
        year: -250, discoverer: "Archimedes"
    },
    // --- Connects: radius, area ---
    {
        id: "sphere_surface",
        name: "Surface Area of a Sphere",
        field: "classical_mechanics",
        equation: "A = 4\\pi R^2",
        description: "Surface area of a sphere — appears in radiation, gravitation, and electrostatics.",
        uses: ["area", "radius"],
        year: -250, discoverer: "Archimedes"
    },
    // --- Connects: reynolds_number, velocity ---
    {
        id: "turbulent_transition",
        name: "Critical Reynolds Number",
        field: "fluid_mechanics",
        equation: "Re_{crit} \\approx 2300",
        description: "Flow transitions from laminar to turbulent at approximately Re = 2300 for pipe flow.",
        uses: ["var_reynolds", "velocity", "density", "viscosity", "distance"],
        year: 1883, discoverer: "Osborne Reynolds"
    },
    // --- Connects: molar_absorptivity, absorbance, path_length, concentration ---
    {
        id: "beer_lambert_transmittance",
        name: "Beer-Lambert Transmittance",
        field: "chemistry",
        equation: "T = 10^{-\\varepsilon \\ell c}",
        description: "Fraction of light transmitted through a solution relates exponentially to absorptivity, path length, and concentration.",
        uses: ["molar_absorptivity", "path_length", "concentration", "intensity"],
        year: 1852, discoverer: "August Beer"
    },

    // ═════════════════════════════════════════════════════════════════════════
    // ACCURACY & CALCULUS — New equations for supersession chains
    // ═════════════════════════════════════════════════════════════════════════

    {
        id: "galilean_velocity_addition",
        name: "Galilean Velocity Addition",
        field: "classical_mechanics",
        equation: "v = v_1 + v_2",
        description: "Classical rule: simply add velocities. Intuitive but incorrect — breaks down near the speed of light, violating the constancy of c.",
        uses: ["velocity"],
        year: 1632, discoverer: "Galileo Galilei"
    },
    {
        id: "relativistic_velocity_addition",
        name: "Relativistic Velocity Addition",
        field: "special_relativity",
        equation: "v = \\frac{v_1 + v_2}{1 + \\dfrac{v_1 v_2}{c^2}}",
        description: "The correct velocity addition law. No matter how fast two objects move, their combined velocity never exceeds c. Reduces to v₁+v₂ when both velocities are much less than c.",
        uses: ["velocity", "const_c"],
        year: 1905, discoverer: "Albert Einstein"
    },
    {
        id: "newton_second_momentum",
        name: "Newton's Second Law (Momentum)",
        field: "classical_mechanics",
        equation: "\\vec{F} = \\frac{d\\vec{p}}{dt}",
        description: "The more general form of Newton's second law — force equals rate of change of momentum. Handles variable-mass systems (rockets) and is the correct bridge to relativistic mechanics.",
        uses: ["force", "momentum", "time"],
        year: 1687, discoverer: "Isaac Newton"
    },
    {
        id: "navier_stokes",
        name: "Navier-Stokes Equation",
        field: "fluid_mechanics",
        equation: "\\rho\\!\\left(\\frac{\\partial \\vec{v}}{\\partial t} + \\vec{v}\\!\\cdot\\!\\nabla\\vec{v}\\right) = -\\nabla P + \\mu\\nabla^2\\vec{v} + \\rho\\vec{g}",
        description: "Fundamental equation of viscous fluid motion — one of the Clay Millennium Prize Problems. Generalizes Bernoulli and Euler equations to include viscosity.",
        uses: ["density", "velocity", "time", "pressure", "viscosity", "const_g"],
        year: 1845, discoverer: "Claude-Louis Navier & George Gabriel Stokes"
    },

    // ═════════════════════════════════════════════════════════════════════════
    // PHASE 3 — Major Expansion
    // ═════════════════════════════════════════════════════════════════════════

    // ─── ELECTROMAGNETISM (completing Maxwell's equations) ────────────────────

    {
        id: "ampere_maxwell",
        name: "Ampère-Maxwell Law",
        field: "electromagnetism",
        equation: "\\nabla \\times \\vec{B} = \\mu_0\\vec{J} + \\mu_0\\varepsilon_0\\frac{\\partial \\vec{E}}{\\partial t}",
        description: "Fourth of Maxwell's equations. Magnetic fields are generated by currents and changing electric fields — the displacement current that completed classical electrodynamics.",
        uses: ["magnetic_field", "const_mu0", "current", "const_eps0", "electric_field", "time"],
        year: 1865, discoverer: "James Clerk Maxwell"
    },
    {
        id: "gauss_magnetism",
        name: "Gauss's Law for Magnetism",
        field: "electromagnetism",
        equation: "\\nabla \\cdot \\vec{B} = 0",
        description: "Second of Maxwell's equations. There are no magnetic monopoles — every magnetic field line forms a closed loop.",
        uses: ["magnetic_field"],
        year: 1835, discoverer: "Carl Friedrich Gauss"
    },
    {
        id: "poynting_vector",
        name: "Poynting Vector",
        field: "electromagnetism",
        equation: "\\vec{S} = \\frac{1}{\\mu_0}\\vec{E} \\times \\vec{B}",
        description: "Directional energy flux of an electromagnetic field — power per unit area carried by EM waves.",
        uses: ["power", "area", "electric_field", "magnetic_field", "const_mu0"],
        year: 1884, discoverer: "John Henry Poynting"
    },
    {
        id: "em_wave_equation",
        name: "EM Wave Equation",
        field: "electromagnetism",
        equation: "\\nabla^2 \\vec{E} = \\mu_0\\varepsilon_0 \\frac{\\partial^2 \\vec{E}}{\\partial t^2}",
        description: "Electromagnetic waves propagate at the speed of light — derived from Maxwell's equations, unifying optics and electromagnetism.",
        uses: ["electric_field", "const_mu0", "const_eps0", "time"],
        year: 1865, discoverer: "James Clerk Maxwell"
    },
    {
        id: "solenoid_field",
        name: "Solenoid Magnetic Field",
        field: "electromagnetism",
        equation: "B = \\mu_0 n I",
        description: "Uniform magnetic field inside an ideal solenoid.",
        uses: ["magnetic_field", "const_mu0", "current"],
        year: 1820, discoverer: "André-Marie Ampère"
    },
    {
        id: "lc_resonance",
        name: "LC Circuit Resonance",
        field: "electromagnetism",
        equation: "\\omega_0 = \\frac{1}{\\sqrt{LC}}",
        description: "Natural angular frequency of an LC oscillator — electrical analog of a mass-spring system.",
        uses: ["angular_frequency", "inductance", "capacitance"],
        year: 1853, discoverer: "William Thomson (Lord Kelvin)"
    },
    {
        id: "cyclotron_frequency",
        name: "Cyclotron Frequency",
        field: "electromagnetism",
        equation: "\\omega_c = \\frac{qB}{m}",
        description: "Angular frequency of a charged particle spiraling in a magnetic field — basis of cyclotrons and mass spectrometers.",
        uses: ["angular_frequency", "charge", "magnetic_field", "mass"],
        year: 1932, discoverer: "Ernest Lawrence"
    },

    // ─── CLASSICAL MECHANICS (Lagrangian/Hamiltonian) ────────────────────────

    {
        id: "euler_lagrange",
        name: "Euler-Lagrange Equation",
        field: "classical_mechanics",
        equation: "\\frac{d}{dt}\\frac{\\partial \\mathcal{L}}{\\partial \\dot{q}} - \\frac{\\partial \\mathcal{L}}{\\partial q} = 0",
        description: "The fundamental equation of analytical mechanics. All of classical physics follows from a single variational principle — stationary action.",
        uses: ["energy", "time", "displacement", "velocity"],
        year: 1788, discoverer: "Joseph-Louis Lagrange"
    },
    {
        id: "hamiltons_equations",
        name: "Hamilton's Equations",
        field: "classical_mechanics",
        equation: "\\dot{q} = \\frac{\\partial H}{\\partial p}, \\quad \\dot{p} = -\\frac{\\partial H}{\\partial q}",
        description: "Phase space formulation of mechanics — reveals symplectic geometry and bridges directly to quantum mechanics.",
        uses: ["energy", "momentum", "displacement", "time"],
        year: 1833, discoverer: "William Rowan Hamilton"
    },
    {
        id: "virial_theorem",
        name: "Virial Theorem",
        field: "classical_mechanics",
        equation: "\\langle T \\rangle = -\\frac{1}{2}\\sum_k \\langle \\vec{F}_k \\cdot \\vec{r}_k \\rangle",
        description: "Relates time-averaged kinetic energy to forces — fundamental in astrophysics for estimating masses of galaxies and clusters.",
        uses: ["energy", "force", "distance"],
        year: 1870, discoverer: "Rudolf Clausius"
    },
    {
        id: "elastic_pe",
        name: "Elastic Potential Energy",
        field: "classical_mechanics",
        equation: "U = \\tfrac{1}{2}kx^2",
        description: "Potential energy stored in a deformed elastic body.",
        uses: ["energy", "spring_constant", "displacement"],
        year: 1678, discoverer: "Robert Hooke"
    },
    {
        id: "impulse_momentum",
        name: "Impulse-Momentum Theorem",
        field: "classical_mechanics",
        equation: "\\vec{J} = \\int \\vec{F}\\,dt = \\Delta \\vec{p}",
        description: "The impulse (time-integrated force) equals the change in momentum.",
        uses: ["force", "time", "momentum"],
        year: 1687, discoverer: "Isaac Newton"
    },
    {
        id: "torricelli",
        name: "Torricelli's Theorem",
        field: "fluid_mechanics",
        equation: "v = \\sqrt{2gh}",
        description: "Speed of fluid flowing from an orifice under gravity — a direct consequence of Bernoulli's equation.",
        uses: ["velocity", "const_g", "height"],
        year: 1643, discoverer: "Evangelista Torricelli"
    },

    // ─── QUANTUM MECHANICS ───────────────────────────────────────────────────

    {
        id: "born_rule",
        name: "Born Rule",
        field: "quantum_mechanics",
        equation: "P(x) = |\\psi(x)|^2",
        description: "The probability of finding a particle at position x is the squared modulus of the wave function — the interpretive heart of quantum mechanics.",
        uses: ["probability", "wave_function"],
        year: 1926, discoverer: "Max Born"
    },
    {
        id: "commutation_xp",
        name: "Canonical Commutation Relation",
        field: "quantum_mechanics",
        equation: "[\\hat{x}, \\hat{p}] = i\\hbar",
        description: "Position and momentum operators do not commute — the algebraic foundation of the uncertainty principle.",
        uses: ["displacement", "momentum", "const_hbar"],
        year: 1925, discoverer: "Werner Heisenberg"
    },
    {
        id: "particle_in_box",
        name: "Particle in a Box",
        field: "quantum_mechanics",
        equation: "E_n = \\frac{n^2 \\pi^2 \\hbar^2}{2mL^2}",
        description: "Quantized energy levels of a particle confined to an infinite potential well — the simplest quantum system.",
        uses: ["energy", "const_hbar", "mass", "length", "principal_quantum"],
        year: 1926, discoverer: "Erwin Schrödinger"
    },
    {
        id: "qm_harmonic_oscillator",
        name: "Quantum Harmonic Oscillator",
        field: "quantum_mechanics",
        equation: "E_n = \\left(n + \\tfrac{1}{2}\\right)\\hbar\\omega",
        description: "Quantized energy levels of a harmonic oscillator — includes zero-point energy ½ℏω even at absolute zero.",
        uses: ["energy", "const_hbar", "angular_frequency", "principal_quantum"],
        year: 1926, discoverer: "Erwin Schrödinger"
    },
    {
        id: "dirac_equation",
        name: "Dirac Equation",
        field: "quantum_mechanics",
        equation: "(i\\gamma^\\mu \\partial_\\mu - \\frac{mc}{\\hbar})\\psi = 0",
        description: "Relativistic wave equation for spin-½ particles. Predicted the positron and the intrinsic spin and magnetic moment of the electron.",
        uses: ["wave_function", "mass", "const_c", "const_hbar"],
        year: 1928, discoverer: "Paul Dirac"
    },
    {
        id: "fermis_golden_rule",
        name: "Fermi's Golden Rule",
        field: "quantum_mechanics",
        equation: "\\Gamma_{i \\to f} = \\frac{2\\pi}{\\hbar}|\\langle f|\\hat{H}'|i\\rangle|^2 \\rho(E_f)",
        description: "Transition rate between quantum states under a perturbation — fundamental to scattering, decay, and spectroscopy.",
        uses: ["const_hbar", "energy", "probability"],
        year: 1950, discoverer: "Enrico Fermi"
    },
    {
        id: "tunnel_transmission",
        name: "Quantum Tunneling",
        field: "quantum_mechanics",
        equation: "T \\approx e^{-2\\kappa L}, \\quad \\kappa = \\sqrt{\\frac{2m(V-E)}{\\hbar^2}}",
        description: "A particle can pass through a potential barrier it classically cannot surmount — transmission probability decays exponentially with barrier width.",
        uses: ["mass", "energy", "const_hbar", "length"],
        year: 1928, discoverer: "George Gamow"
    },
    {
        id: "ehrenfest_theorem",
        name: "Ehrenfest Theorem",
        field: "quantum_mechanics",
        equation: "m\\frac{d\\langle x \\rangle}{dt} = \\langle p \\rangle",
        description: "Quantum expectation values obey classical equations of motion — the bridge between quantum and classical physics.",
        uses: ["mass", "displacement", "time", "momentum"],
        year: 1927, discoverer: "Paul Ehrenfest"
    },
    {
        id: "klein_gordon",
        name: "Klein-Gordon Equation",
        field: "quantum_mechanics",
        equation: "\\left(\\partial^\\mu \\partial_\\mu + \\frac{m^2c^2}{\\hbar^2}\\right)\\phi = 0",
        description: "Relativistic wave equation for spin-0 particles. First attempt at relativistic QM — predates and is simpler than the Dirac equation.",
        uses: ["mass", "const_c", "const_hbar", "wave_function"],
        year: 1926, discoverer: "Oskar Klein & Walter Gordon"
    },
    {
        id: "spin_angular_momentum",
        name: "Spin Angular Momentum",
        field: "quantum_mechanics",
        equation: "S = \\sqrt{s(s+1)}\\,\\hbar",
        description: "Intrinsic angular momentum of a particle is quantized in units of ℏ.",
        uses: ["angular_momentum", "const_hbar"],
        year: 1925, discoverer: "George Uhlenbeck & Samuel Goudsmit"
    },

    // ─── SPECIAL RELATIVITY ──────────────────────────────────────────────────

    {
        id: "lorentz_transformation",
        name: "Lorentz Transformation",
        field: "special_relativity",
        equation: "x' = \\gamma(x - vt), \\quad t' = \\gamma\\!\\left(t - \\frac{vx}{c^2}\\right)",
        description: "Coordinate transformations between inertial frames — the foundation of special relativity, replacing Galilean transformations.",
        uses: ["displacement", "velocity", "time", "lorentz_factor", "const_c"],
        year: 1904, discoverer: "Hendrik Lorentz"
    },
    {
        id: "spacetime_interval",
        name: "Invariant Spacetime Interval",
        field: "special_relativity",
        equation: "ds^2 = -c^2 dt^2 + dx^2 + dy^2 + dz^2",
        description: "The spacetime interval is invariant under Lorentz transformations — replaces absolute time and absolute space with absolute spacetime.",
        uses: ["displacement", "time", "const_c"],
        year: 1908, discoverer: "Hermann Minkowski"
    },
    {
        id: "four_momentum",
        name: "Four-Momentum",
        field: "special_relativity",
        equation: "p^\\mu = \\left(\\frac{E}{c}, \\vec{p}\\right)",
        description: "Energy and momentum unified into a single four-vector — the natural language of relativistic mechanics.",
        uses: ["energy", "const_c", "momentum"],
        year: 1905, discoverer: "Albert Einstein"
    },

    // ─── GENERAL RELATIVITY ──────────────────────────────────────────────────

    {
        id: "schwarzschild_metric",
        name: "Schwarzschild Metric",
        field: "general_relativity",
        equation: "ds^2 = -\\!\\left(1 - \\frac{r_s}{r}\\right)c^2 dt^2 + \\frac{dr^2}{1 - r_s/r} + r^2 d\\Omega^2",
        description: "Exact spacetime geometry outside a spherically symmetric, non-rotating mass — describes black holes and planetary orbits.",
        uses: ["distance", "time", "const_c", "mass", "const_G", "metric_tensor"],
        year: 1916, discoverer: "Karl Schwarzschild"
    },
    {
        id: "gravitational_lensing",
        name: "Gravitational Lensing Angle",
        field: "general_relativity",
        equation: "\\Delta\\phi = \\frac{4GM}{rc^2}",
        description: "Light bends around massive objects by twice the Newtonian prediction — the 1919 eclipse observation that confirmed GR.",
        uses: ["angle", "const_G", "mass", "distance", "const_c"],
        year: 1915, discoverer: "Albert Einstein"
    },
    {
        id: "stress_energy_conservation",
        name: "Stress-Energy Conservation",
        field: "general_relativity",
        equation: "\\nabla_\\mu T^{\\mu\\nu} = 0",
        description: "Local conservation of energy and momentum in curved spacetime — the GR generalization of Noether's theorem for translation invariance.",
        uses: ["stress_energy_tensor"],
        year: 1915, discoverer: "Albert Einstein"
    },
    {
        id: "cosmological_redshift",
        name: "Cosmological Redshift",
        field: "astrophysics",
        equation: "1 + z = \\frac{a_0}{a(t_{\\text{emit}})}",
        description: "The wavelength of light stretches with the expansion of the universe — distinct from Doppler shift.",
        uses: ["scale_factor", "redshift"],
        year: 1922, discoverer: "Alexander Friedmann"
    },

    // ─── THERMODYNAMICS & STATISTICAL MECHANICS ──────────────────────────────

    {
        id: "van_der_waals",
        name: "Van der Waals Equation",
        field: "thermodynamics",
        equation: "\\left(P + \\frac{a}{V^2}\\right)(V - b) = nRT",
        description: "Equation of state for real gases — corrects for intermolecular forces and finite molecular volume.",
        uses: ["pressure", "volume", "moles", "const_R", "temperature"],
        year: 1873, discoverer: "Johannes Diderik van der Waals"
    },
    {
        id: "clausius_inequality",
        name: "Clausius Inequality",
        field: "thermodynamics",
        equation: "\\oint \\frac{\\delta Q}{T} \\leq 0",
        description: "The mathematical statement of the Second Law — entropy of the universe never decreases.",
        uses: ["heat", "temperature", "entropy"],
        year: 1854, discoverer: "Rudolf Clausius"
    },
    {
        id: "maxwell_relation_tv",
        name: "Maxwell Relation (T-V)",
        field: "thermodynamics",
        equation: "\\left(\\frac{\\partial T}{\\partial V}\\right)_S = -\\left(\\frac{\\partial P}{\\partial S}\\right)_V",
        description: "One of four Maxwell relations — equalities between mixed partial derivatives of thermodynamic potentials.",
        uses: ["temperature", "volume", "entropy", "pressure"],
        year: 1871, discoverer: "James Clerk Maxwell"
    },
    {
        id: "adiabatic_process",
        name: "Adiabatic Process",
        field: "thermodynamics",
        equation: "PV^\\gamma = \\text{const}",
        description: "Pressure-volume relation for a reversible adiabatic process — no heat exchange with surroundings.",
        uses: ["pressure", "volume", "heat_capacity_ratio"],
        year: 1823, discoverer: "Siméon Denis Poisson"
    },
    {
        id: "third_law_thermo",
        name: "Third Law (Nernst Theorem)",
        field: "thermodynamics",
        equation: "\\lim_{T \\to 0} S = 0",
        description: "Entropy approaches zero as temperature approaches absolute zero — it's impossible to reach T = 0 in finite steps.",
        uses: ["temperature", "entropy"],
        year: 1906, discoverer: "Walther Nernst"
    },
    {
        id: "grand_canonical",
        name: "Grand Canonical Partition Function",
        field: "statistical_mechanics",
        equation: "\\mathcal{Z} = \\sum_{N} \\sum_i e^{-\\beta(E_i - \\mu N)}",
        description: "Partition function for systems exchanging both energy and particles — foundation of open-system thermodynamics.",
        uses: ["var_partition_func", "energy", "chemical_potential", "num_particles", "const_kB", "temperature"],
        year: 1902, discoverer: "Josiah Willard Gibbs"
    },
    {
        id: "fluctuation_dissipation",
        name: "Fluctuation-Dissipation Theorem",
        field: "statistical_mechanics",
        equation: "\\langle (\\Delta E)^2 \\rangle = k_B T^2 C_V",
        description: "Connects spontaneous fluctuations to dissipative response — a deep bridge between equilibrium and near-equilibrium physics.",
        uses: ["energy", "const_kB", "temperature", "specific_heat"],
        year: 1951, discoverer: "Herbert Callen & Theodore Welton"
    },
    {
        id: "mean_free_path_def",
        name: "Mean Free Path",
        field: "statistical_mechanics",
        equation: "\\lambda = \\frac{1}{n\\sigma}",
        description: "Average distance a particle travels between collisions — bridges microscopic cross-sections to macroscopic transport.",
        uses: ["mean_free_path", "num_particles", "cross_section"],
        year: 1860, discoverer: "James Clerk Maxwell"
    },
    {
        id: "nyquist_johnson",
        name: "Nyquist-Johnson Noise",
        field: "statistical_mechanics",
        equation: "\\langle V^2 \\rangle = 4k_B T R \\Delta f",
        description: "Thermal voltage noise across a resistor — a direct manifestation of the fluctuation-dissipation theorem in electronics.",
        uses: ["const_kB", "temperature", "resistance", "bandwidth"],
        year: 1928, discoverer: "Harry Nyquist & John B. Johnson"
    },

    // ─── OPTICS & WAVES ──────────────────────────────────────────────────────

    {
        id: "brewster_angle",
        name: "Brewster's Angle",
        field: "optics_waves",
        equation: "\\tan\\theta_B = \\frac{n_2}{n_1}",
        description: "Angle of incidence at which reflected light is perfectly polarized — no reflected p-polarized component.",
        uses: ["angle", "refractive_index"],
        year: 1815, discoverer: "David Brewster"
    },
    {
        id: "rayleigh_criterion",
        name: "Rayleigh Criterion",
        field: "optics_waves",
        equation: "\\theta_{min} = 1.22\\frac{\\lambda}{D}",
        description: "Angular resolution limit of a circular aperture — sets the fundamental limit of telescopes and microscopes.",
        uses: ["angle", "wavelength", "distance"],
        year: 1879, discoverer: "Lord Rayleigh"
    },
    {
        id: "double_slit",
        name: "Young's Double Slit",
        field: "optics_waves",
        equation: "\\Delta = d\\sin\\theta = m\\lambda",
        description: "Condition for constructive interference from two slits — proved the wave nature of light.",
        uses: ["distance", "angle", "wavelength"],
        year: 1801, discoverer: "Thomas Young"
    },
    {
        id: "lensmaker",
        name: "Lensmaker's Equation",
        field: "optics_waves",
        equation: "\\frac{1}{f} = (n-1)\\!\\left(\\frac{1}{R_1} - \\frac{1}{R_2}\\right)",
        description: "Focal length of a thin lens in terms of its curvatures and refractive index.",
        uses: ["distance", "refractive_index", "radius"],
        year: 1693, discoverer: "Edmond Halley"
    },
    {
        id: "total_internal_reflection",
        name: "Total Internal Reflection",
        field: "optics_waves",
        equation: "\\sin\\theta_c = \\frac{n_2}{n_1}",
        description: "Critical angle beyond which light is totally reflected — the principle behind fiber optics.",
        uses: ["angle", "refractive_index"],
        year: 1621, discoverer: "Willebrord Snellius"
    },
    {
        id: "bragg_diffraction",
        name: "Bragg's Law",
        field: "optics_waves",
        equation: "n\\lambda = 2d\\sin\\theta",
        description: "Condition for constructive interference from crystal planes — the basis of X-ray crystallography and materials science.",
        uses: ["wavelength", "distance", "angle"],
        year: 1913, discoverer: "William Henry Bragg & William Lawrence Bragg"
    },

    // ─── NUCLEAR & PARTICLE PHYSICS ──────────────────────────────────────────

    {
        id: "bethe_weizsacker",
        name: "Semi-Empirical Mass Formula",
        field: "nuclear_physics",
        equation: "B = a_V A - a_S A^{2/3} - a_C \\frac{Z(Z-1)}{A^{1/3}} - a_A \\frac{(A-2Z)^2}{A} + \\delta",
        description: "Liquid drop model of the nucleus — predicts binding energies by accounting for volume, surface, Coulomb, asymmetry, and pairing terms.",
        uses: ["energy", "mass_number", "atomic_number"],
        year: 1935, discoverer: "Carl Friedrich von Weizsäcker"
    },
    {
        id: "q_value",
        name: "Nuclear Q-Value",
        field: "nuclear_physics",
        equation: "Q = (\\Sigma m_i - \\Sigma m_f) c^2",
        description: "Energy released or absorbed in a nuclear reaction from the mass difference between reactants and products.",
        uses: ["mass", "const_c", "energy"],
        year: 1932, discoverer: "John Cockcroft & Ernest Walton"
    },
    {
        id: "geiger_nuttall",
        name: "Geiger-Nuttall Law",
        field: "nuclear_physics",
        equation: "\\log_{10} t_{1/2} = a + \\frac{b}{\\sqrt{E_\\alpha}}",
        description: "Empirical relation between alpha decay half-life and kinetic energy — explained by quantum tunneling through the Coulomb barrier.",
        uses: ["half_life", "energy"],
        year: 1911, discoverer: "Hans Geiger & John Mitchell Nuttall"
    },
    {
        id: "rutherford_scattering",
        name: "Rutherford Scattering",
        field: "nuclear_physics",
        equation: "\\frac{d\\sigma}{d\\Omega} = \\left(\\frac{Z_1 Z_2 e^2}{4E}\\right)^2 \\frac{1}{\\sin^4(\\theta/2)}",
        description: "Differential cross-section for Coulomb scattering — proved the nuclear model of the atom in 1911.",
        uses: ["cross_section", "atomic_number", "const_e", "energy", "angle", "scattering_angle", "solid_angle"],
        year: 1911, discoverer: "Ernest Rutherford"
    },
    {
        id: "cross_section_rate",
        name: "Reaction Rate from Cross-Section",
        field: "nuclear_physics",
        equation: "R = n_1 n_2 \\sigma v",
        description: "Rate of nuclear reactions from beam-target geometry — the fundamental equation of nuclear and particle experiments.",
        uses: ["num_particles", "cross_section", "velocity"],
        year: 1932, discoverer: "Ernest Rutherford"
    },

    // ─── ASTROPHYSICS & COSMOLOGY ────────────────────────────────────────────

    {
        id: "jeans_mass",
        name: "Jeans Mass",
        field: "astrophysics",
        equation: "M_J \\sim \\left(\\frac{k_B T}{G m_p}\\right)^{3/2} \\rho^{-1/2}",
        description: "Minimum mass for gravitational collapse — thermal pressure vs. gravity determines whether a gas cloud forms a star.",
        uses: ["mass", "const_kB", "temperature", "const_G", "const_mp", "density"],
        year: 1902, discoverer: "James Jeans"
    },
    {
        id: "tov_equation",
        name: "Tolman-Oppenheimer-Volkoff Eq.",
        field: "astrophysics",
        equation: "\\frac{dP}{dr} = -\\frac{(\\rho + P/c^2)(m + 4\\pi r^3 P/c^2)}{r^2(1 - 2Gm/rc^2)} G",
        description: "Hydrostatic equilibrium in GR — determines the structure of neutron stars and the maximum neutron star mass.",
        uses: ["pressure", "distance", "const_G", "density", "const_c", "mass"],
        year: 1939, discoverer: "Richard Tolman, Robert Oppenheimer & George Volkoff"
    },
    {
        id: "eddington_luminosity",
        name: "Eddington Luminosity",
        field: "astrophysics",
        equation: "L_E = \\frac{4\\pi G M c}{\\kappa}",
        description: "Maximum luminosity a body can achieve where radiation pressure balances gravity — sets a limit on stellar brightness and black hole accretion.",
        uses: ["luminosity", "const_G", "mass", "const_c"],
        year: 1926, discoverer: "Arthur Eddington"
    },
    {
        id: "saha_equation",
        name: "Saha Ionization Equation",
        field: "astrophysics",
        equation: "\\frac{n_{i+1} n_e}{n_i} = \\frac{2}{\\lambda_{th}^3}\\frac{g_{i+1}}{g_i}e^{-\\chi_i / k_B T}",
        description: "Ionization equilibrium in stellar atmospheres — determines spectral types and stellar classification.",
        uses: ["num_particles", "const_kB", "temperature", "const_h", "mass"],
        year: 1920, discoverer: "Meghnad Saha"
    },

    // ─── CHEMISTRY ───────────────────────────────────────────────────────────

    {
        id: "henderson_hasselbalch",
        name: "Henderson-Hasselbalch Eq.",
        field: "chemistry",
        equation: "\\text{pH} = \\text{p}K_a + \\log_{10}\\frac{[\\text{A}^-]}{[\\text{HA}]}",
        description: "Relates pH of a buffer solution to the pKa and the ratio of conjugate base to acid — essential in biochemistry.",
        uses: ["concentration"],
        year: 1908, discoverer: "Lawrence Joseph Henderson"
    },
    {
        id: "hesss_law",
        name: "Hess's Law",
        field: "chemistry",
        equation: "\\Delta H_{rxn} = \\sum \\Delta H_f^\\circ(\\text{products}) - \\sum \\Delta H_f^\\circ(\\text{reactants})",
        description: "Total enthalpy change is path-independent — a direct consequence of enthalpy being a state function.",
        uses: ["enthalpy"],
        year: 1840, discoverer: "Germain Henri Hess"
    },
    {
        id: "grahams_law",
        name: "Graham's Law of Effusion",
        field: "chemistry",
        equation: "\\frac{r_1}{r_2} = \\sqrt{\\frac{M_2}{M_1}}",
        description: "Rate of effusion of a gas is inversely proportional to the square root of its molar mass.",
        uses: ["velocity", "molar_mass"],
        year: 1848, discoverer: "Thomas Graham"
    },
    {
        id: "henry_law",
        name: "Henry's Law",
        field: "chemistry",
        equation: "P = k_H \\, c",
        description: "Concentration of dissolved gas is proportional to its partial pressure — governs CO₂ in oceans and O₂ in blood.",
        uses: ["pressure", "concentration"],
        year: 1803, discoverer: "William Henry"
    },

    // ─── FLUID MECHANICS ─────────────────────────────────────────────────────

    {
        id: "young_laplace",
        name: "Young-Laplace Equation",
        field: "fluid_mechanics",
        equation: "\\Delta P = \\gamma\\!\\left(\\frac{1}{R_1} + \\frac{1}{R_2}\\right)",
        description: "Pressure difference across a curved interface due to surface tension — governs bubbles, droplets, and capillary action.",
        uses: ["pressure", "surface_tension", "radius"],
        year: 1805, discoverer: "Thomas Young & Pierre-Simon Laplace"
    },
    {
        id: "stokes_einstein",
        name: "Stokes-Einstein Diffusion",
        field: "fluid_mechanics",
        equation: "D = \\frac{k_B T}{6\\pi \\mu r}",
        description: "Diffusion coefficient of a spherical particle — bridges thermodynamics, fluid mechanics, and Brownian motion.",
        uses: ["diffusion_coefficient", "const_kB", "temperature", "viscosity", "distance"],
        year: 1905, discoverer: "Albert Einstein"
    },
    {
        id: "fick_first_law",
        name: "Fick's First Law of Diffusion",
        field: "fluid_mechanics",
        equation: "\\vec{J} = -D\\nabla c",
        description: "Diffusive flux is proportional to the negative concentration gradient — the diffusion analog of Fourier's law.",
        uses: ["diffusion_coefficient", "concentration"],
        year: 1855, discoverer: "Adolf Fick"
    },

    // ─── INFORMATION THEORY ──────────────────────────────────────────────────

    {
        id: "shannon_hartley",
        name: "Shannon-Hartley Channel Capacity",
        field: "information_theory",
        equation: "C = B \\log_2\\!\\left(1 + \\frac{S}{N}\\right)",
        description: "Maximum error-free data rate over a noisy channel — the fundamental theorem of digital communication.",
        uses: ["channel_capacity", "bandwidth", "signal_noise_ratio"],
        year: 1948, discoverer: "Claude Shannon"
    },
    {
        id: "kl_divergence",
        name: "Kullback-Leibler Divergence",
        field: "information_theory",
        equation: "D_{KL}(P \\| Q) = \\sum_i p_i \\ln \\frac{p_i}{q_i}",
        description: "Measures how one probability distribution diverges from a reference — foundation of machine learning and Bayesian inference.",
        uses: ["probability", "information_entropy"],
        year: 1951, discoverer: "Solomon Kullback & Richard Leibler"
    },
    {
        id: "mutual_information",
        name: "Mutual Information",
        field: "information_theory",
        equation: "I(X;Y) = H(X) - H(X|Y)",
        description: "Quantifies the information one random variable contains about another — generalizes correlation to nonlinear relationships.",
        uses: ["information_entropy"],
        year: 1948, discoverer: "Claude Shannon"
    },
    {
        id: "data_processing_inequality",
        name: "Data Processing Inequality",
        field: "information_theory",
        equation: "I(X;Y) \\geq I(X;Z) \\quad \\text{if } X \\to Y \\to Z",
        description: "Processing data can only destroy information, never create it — a fundamental constraint on all computation and inference.",
        uses: ["information_entropy"],
        year: 1961, discoverer: "Solomon Kullback"
    },

    // ─── CROSS-FIELD BRIDGES ─────────────────────────────────────────────────

    {
        id: "einstein_diffusion",
        name: "Einstein Relation (Mobility)",
        field: "statistical_mechanics",
        equation: "D = \\mu k_B T",
        description: "Connects diffusion coefficient to particle mobility and temperature — a key bridge between statistical mechanics and transport.",
        uses: ["diffusion_coefficient", "const_kB", "temperature"],
        year: 1905, discoverer: "Albert Einstein"
    },
    {
        id: "magnetic_moment_orbital",
        name: "Orbital Magnetic Moment",
        field: "quantum_mechanics",
        equation: "\\vec{\\mu}_L = -\\frac{e}{2m_e}\\vec{L}",
        description: "The magnetic moment of an orbiting electron — connects electromagnetism to quantum angular momentum.",
        uses: ["magnetic_moment", "const_e", "const_me", "angular_momentum"],
        year: 1920, discoverer: "Alfred Landé"
    },
    {
        id: "mass_density",
        name: "Mass-Volume Density",
        field: "classical_mechanics",
        equation: "\\rho = \\frac{m}{V}",
        description: "Density is mass per unit volume — the most basic bridge between mechanics and fluid/material properties.",
        uses: ["density", "mass", "volume"],
        year: -250, discoverer: "Archimedes"
    },
    {
        id: "ideal_gas_kinetic",
        name: "Kinetic Theory of Temperature",
        field: "thermodynamics",
        equation: "\\frac{1}{2}m\\langle v^2 \\rangle = \\frac{3}{2}k_B T",
        description: "Temperature IS average kinetic energy — the deepest bridge between mechanics and thermodynamics.",
        uses: ["mass", "velocity", "const_kB", "temperature"],
        year: 1857, discoverer: "Rudolf Clausius"
    },
    {
        id: "dipole_field",
        name: "Electric Dipole Field",
        field: "electromagnetism",
        equation: "E \\approx \\frac{1}{4\\pi\\varepsilon_0}\\frac{p}{r^3}",
        description: "Far-field of an electric dipole falls off as 1/r³ — governs molecular interactions and dielectric behavior.",
        uses: ["electric_field", "const_eps0", "electric_dipole", "distance"],
        year: 1785, discoverer: "Charles-Augustin de Coulomb"
    },
    {
        id: "molar_mass_relation",
        name: "Molar Mass Relation",
        field: "chemistry",
        equation: "M = N_A \\cdot m_{\\text{particle}}",
        description: "Molar mass equals Avogadro's number times the mass of one particle — connects the molecular and macroscopic worlds.",
        uses: ["molar_mass", "const_NA", "mass"],
        year: 1811, discoverer: "Amedeo Avogadro"
    },

    // ═════════════════════════════════════════════════════════════════════════
    // NEW EQUATIONS — Batch 2 (60 additional)
    // ═════════════════════════════════════════════════════════════════════════

    // ─── Classical Mechanics ────────────────────────────────────────────────

    {
        id: "hookes_law_stress_strain",
        name: "Hooke's Law (Stress–Strain)",
        field: "classical_mechanics",
        equation: "\\sigma = E \\varepsilon",
        description: "Within the elastic limit, stress is proportional to strain — the continuum version of Hooke's law underlying all structural engineering.",
        uses: ["stress", "youngs_modulus", "strain"],
        year: 1678, discoverer: "Robert Hooke"
    },
    {
        id: "bulk_modulus_def",
        name: "Bulk Modulus",
        field: "classical_mechanics",
        equation: "K = -V \\frac{dP}{dV}",
        description: "Resistance of a substance to uniform compression — relates pressure change to fractional volume change.",
        uses: ["bulk_modulus", "volume", "pressure"],
        year: 1694, discoverer: "Robert Boyle"
    },
    {
        id: "poisson_ratio_def",
        name: "Poisson's Ratio",
        field: "classical_mechanics",
        equation: "\\nu = -\\frac{\\varepsilon_{\\text{lateral}}}{\\varepsilon_{\\text{axial}}}",
        description: "The ratio of transverse to axial strain — characterizes how materials deform laterally when stretched.",
        uses: ["poisson_ratio", "strain"],
        year: 1827, discoverer: "Siméon Denis Poisson"
    },
    {
        id: "parallel_axis_theorem",
        name: "Parallel Axis Theorem",
        field: "classical_mechanics",
        equation: "I = I_{\\text{cm}} + Md^2",
        description: "Moment of inertia about any axis equals the center-of-mass moment plus mass times the squared distance between axes.",
        uses: ["moment_of_inertia", "mass", "distance"],
        year: 1749, discoverer: "Christiaan Huygens"
    },
    {
        id: "damped_oscillator",
        name: "Damped Harmonic Oscillator",
        field: "classical_mechanics",
        equation: "m\\ddot{x} + b\\dot{x} + kx = 0",
        description: "A harmonic oscillator with friction — models everything from car suspensions to atomic vibrations in dissipative environments.",
        uses: ["mass", "displacement", "spring_constant", "velocity"],
        year: 1743, discoverer: "Jean le Rond d'Alembert"
    },
    {
        id: "reduced_mass",
        name: "Reduced Mass",
        field: "classical_mechanics",
        equation: "\\mu = \\frac{m_1 m_2}{m_1 + m_2}",
        description: "Converts a two-body problem into an equivalent one-body problem — essential in orbital mechanics and molecular physics.",
        uses: ["mass"],
        year: 1687, discoverer: "Isaac Newton"
    },

    // ─── Electromagnetism ───────────────────────────────────────────────────

    {
        id: "maxwell_displacement_current",
        name: "Displacement Current",
        field: "electromagnetism",
        equation: "\\vec{J}_d = \\varepsilon_0 \\frac{\\partial \\vec{E}}{\\partial t}",
        description: "Maxwell's revolutionary addition — a changing electric field acts as a current source, completing the symmetry of electromagnetism and enabling electromagnetic waves.",
        uses: ["current_density", "const_eps0", "electric_field", "time"],
        year: 1865, discoverer: "James Clerk Maxwell"
    },
    {
        id: "coulombs_law_vector",
        name: "Coulomb's Law (Vector)",
        field: "electromagnetism",
        equation: "\\vec{F} = \\frac{1}{4\\pi\\varepsilon_0}\\frac{q_1 q_2}{r^2}\\hat{r}",
        description: "The full vector form of Coulomb's law — specifies both magnitude and direction of the electrostatic force between point charges.",
        uses: ["force", "const_eps0", "charge", "distance"],
        year: 1785, discoverer: "Charles-Augustin de Coulomb"
    },
    {
        id: "ohms_law_field",
        name: "Ohm's Law (Field Form)",
        field: "electromagnetism",
        equation: "\\vec{J} = \\sigma \\vec{E}",
        description: "The local, microscopic form of Ohm's law — current density is proportional to electric field, with conductivity as the constant.",
        uses: ["current_density", "electric_field", "resistivity_var"],
        year: 1827, discoverer: "Georg Ohm"
    },
    {
        id: "polarization_field",
        name: "Polarization in Dielectrics",
        field: "electromagnetism",
        equation: "\\vec{P} = \\varepsilon_0 \\chi_e \\vec{E}",
        description: "How a dielectric material responds to an applied electric field — the induced polarization is proportional to the field through the electric susceptibility.",
        uses: ["polarization", "const_eps0", "electric_susceptibility", "electric_field"],
        year: 1837, discoverer: "Michael Faraday"
    },
    {
        id: "energy_density_em",
        name: "EM Energy Density",
        field: "electromagnetism",
        equation: "u = \\frac{1}{2}\\varepsilon_0 E^2 + \\frac{1}{2\\mu_0} B^2",
        description: "Energy stored per unit volume in electric and magnetic fields — the fundamental measure of energy carried by electromagnetic waves.",
        uses: ["energy", "const_eps0", "electric_field", "const_mu0", "magnetic_field", "volume"],
        year: 1865, discoverer: "James Clerk Maxwell"
    },
    {
        id: "rc_time_constant",
        name: "RC Time Constant",
        field: "electromagnetism",
        equation: "\\tau = RC",
        description: "The characteristic time scale for charging or discharging a capacitor through a resistor — governs all RC circuit transients.",
        uses: ["time", "resistance", "capacitance"],
        year: 1827, discoverer: "Georg Ohm"
    },

    // ─── Quantum Mechanics ──────────────────────────────────────────────────

    {
        id: "pauli_exclusion",
        name: "Pauli Exclusion Principle",
        field: "quantum_mechanics",
        equation: "\\psi(x_1, x_2) = -\\psi(x_2, x_1)",
        description: "No two identical fermions can occupy the same quantum state — the principle that gives structure to the periodic table and prevents matter from collapsing.",
        uses: ["wave_function", "displacement"],
        year: 1925, discoverer: "Wolfgang Pauli"
    },
    {
        id: "angular_momentum_quantization",
        name: "Angular Momentum Quantization",
        field: "quantum_mechanics",
        equation: "L = \\sqrt{l(l+1)}\\hbar",
        description: "Orbital angular momentum is quantized in units of ℏ — a direct consequence of the rotational symmetry of the Schrödinger equation.",
        uses: ["angular_momentum", "orbital_quantum", "const_hbar"],
        year: 1925, discoverer: "Werner Heisenberg"
    },
    {
        id: "spin_z_quantization",
        name: "Spin-z Quantization",
        field: "quantum_mechanics",
        equation: "S_z = m_s \\hbar",
        description: "The z-component of spin angular momentum can only take discrete values — half-integer for fermions, integer for bosons.",
        uses: ["angular_momentum", "spin_quantum", "const_hbar"],
        year: 1925, discoverer: "George Uhlenbeck & Samuel Goudsmit"
    },
    {
        id: "wkb_approximation",
        name: "WKB Approximation",
        field: "quantum_mechanics",
        equation: "\\psi(x) \\approx \\frac{C}{\\sqrt{p(x)}} \\exp\\!\\left(\\frac{i}{\\hbar}\\int p(x)\\,dx\\right)",
        description: "The semiclassical approximation to quantum mechanics — bridges classical trajectories and wave mechanics, essential for tunneling rate calculations.",
        uses: ["wave_function", "momentum", "const_hbar", "displacement"],
        year: 1926, discoverer: "Wentzel, Kramers & Brillouin"
    },
    {
        id: "variational_principle",
        name: "Variational Principle",
        field: "quantum_mechanics",
        equation: "E_0 \\leq \\frac{\\langle \\psi | \\hat{H} | \\psi \\rangle}{\\langle \\psi | \\psi \\rangle}",
        description: "Any trial wave function gives an energy expectation value that is an upper bound to the true ground state energy — the workhorse of approximate quantum mechanics.",
        uses: ["energy", "wave_function"],
        year: 1930, discoverer: "John C. Slater"
    },
    {
        id: "density_of_states_3d",
        name: "Density of States (3D free)",
        field: "quantum_mechanics",
        equation: "g(E) = \\frac{V}{2\\pi^2}\\left(\\frac{2m}{\\hbar^2}\\right)^{3/2} \\sqrt{E}",
        description: "The number of quantum states per unit energy for free particles in 3D — the foundation of Fermi gas theory and solid-state band structure.",
        uses: ["energy", "volume", "mass", "const_hbar"],
        year: 1927, discoverer: "Arnold Sommerfeld"
    },

    // ─── Special & General Relativity ───────────────────────────────────────

    {
        id: "proper_time_def",
        name: "Proper Time",
        field: "special_relativity",
        equation: "d\\tau = dt\\sqrt{1 - v^2/c^2}",
        description: "The time measured by a clock moving with the object — the Lorentz-invariant time that all observers agree on.",
        uses: ["proper_time", "time", "velocity", "const_c"],
        year: 1905, discoverer: "Albert Einstein"
    },
    {
        id: "relativistic_energy_full",
        name: "Relativistic Total Energy",
        field: "special_relativity",
        equation: "E = \\gamma mc^2",
        description: "The total energy of a massive particle including both rest mass and kinetic energy — reduces to E=mc² at rest.",
        uses: ["energy", "lorentz_factor", "mass", "const_c"],
        year: 1905, discoverer: "Albert Einstein"
    },
    {
        id: "kerr_metric",
        name: "Kerr Metric (Boyer–Lindquist)",
        field: "general_relativity",
        equation: "ds^2 = -\\left(1-\\frac{r_s r}{\\Sigma}\\right)c^2 dt^2 + \\frac{\\Sigma}{\\Delta}dr^2 + \\Sigma\\, d\\theta^2 + \\ldots",
        description: "The spacetime geometry around a rotating black hole — generalizes Schwarzschild to include angular momentum, predicting frame dragging.",
        uses: ["distance", "time", "const_c", "const_G", "mass", "angular_momentum"],
        year: 1963, discoverer: "Roy P. Kerr"
    },
    {
        id: "grav_wave_strain",
        name: "Gravitational Wave Strain",
        field: "general_relativity",
        equation: "h \\sim \\frac{4G}{c^4}\\frac{\\ddot{Q}}{r}",
        description: "The amplitude of gravitational waves from a varying mass quadrupole — the signal that LIGO detects from merging black holes.",
        uses: ["const_G", "const_c", "mass", "distance"],
        year: 1918, discoverer: "Albert Einstein"
    },
    {
        id: "frame_dragging",
        name: "Lense–Thirring Precession",
        field: "general_relativity",
        equation: "\\vec{\\Omega}_{\\text{LT}} = \\frac{2GJ}{c^2 r^3}\\hat{r}",
        description: "A rotating mass drags spacetime around it — confirmed by Gravity Probe B, this is a purely general-relativistic effect with no Newtonian counterpart.",
        uses: ["angular_velocity", "const_G", "angular_momentum", "const_c", "distance"],
        year: 1918, discoverer: "Josef Lense & Hans Thirring"
    },

    // ─── Thermodynamics ─────────────────────────────────────────────────────

    {
        id: "joule_thomson",
        name: "Joule–Thomson Coefficient",
        field: "thermodynamics",
        equation: "\\mu_{\\text{JT}} = \\left(\\frac{\\partial T}{\\partial P}\\right)_H",
        description: "How temperature changes with pressure at constant enthalpy — determines whether a gas cools or heats during throttling, essential for refrigeration.",
        uses: ["temperature", "pressure", "enthalpy"],
        year: 1852, discoverer: "James Joule & William Thomson"
    },
    {
        id: "maxwell_relation_sp",
        name: "Maxwell Relation (S,P)",
        field: "thermodynamics",
        equation: "\\left(\\frac{\\partial T}{\\partial P}\\right)_S = \\left(\\frac{\\partial V}{\\partial S}\\right)_P",
        description: "One of the four Maxwell relations — connects temperature-pressure response to volume-entropy response through the exactness of thermodynamic potentials.",
        uses: ["temperature", "pressure", "volume", "entropy"],
        year: 1871, discoverer: "James Clerk Maxwell"
    },
    {
        id: "gibbs_duhem",
        name: "Gibbs–Duhem Equation",
        field: "thermodynamics",
        equation: "S\\,dT - V\\,dP + \\sum_i N_i\\,d\\mu_i = 0",
        description: "Constrains the variations of intensive variables — not all can change independently, linking temperature, pressure, and chemical potentials.",
        uses: ["entropy", "temperature", "volume", "pressure", "num_particles", "chemical_potential"],
        year: 1876, discoverer: "Josiah Willard Gibbs"
    },
    {
        id: "clapeyron_equation",
        name: "Clapeyron Equation",
        field: "thermodynamics",
        equation: "\\frac{dP}{dT} = \\frac{\\Delta H}{T\\,\\Delta V}",
        description: "Exact relation for the slope of a phase boundary in P-T space — the foundation from which Clausius–Clapeyron is approximated.",
        uses: ["pressure", "temperature", "enthalpy", "volume"],
        year: 1834, discoverer: "Benoît Paul Émile Clapeyron"
    },
    {
        id: "ideal_gas_adiabatic",
        name: "Adiabatic Ideal Gas",
        field: "thermodynamics",
        equation: "PV^\\gamma = \\text{const}",
        description: "For an ideal gas undergoing an adiabatic (no heat exchange) process, pressure times volume to the γ power is conserved.",
        uses: ["pressure", "volume", "heat_capacity_ratio"],
        year: 1823, discoverer: "Siméon Denis Poisson"
    },

    // ─── Statistical Mechanics ──────────────────────────────────────────────

    {
        id: "detailed_balance",
        name: "Detailed Balance",
        field: "statistical_mechanics",
        equation: "\\frac{P(i \\to j)}{P(j \\to i)} = e^{-(E_j - E_i)/k_B T}",
        description: "At thermal equilibrium, every elementary process occurs at the same rate as its reverse — the microscopic foundation of equilibrium thermodynamics.",
        uses: ["probability", "energy", "const_kB", "temperature"],
        year: 1872, discoverer: "Ludwig Boltzmann"
    },
    {
        id: "ising_partition",
        name: "1D Ising Partition Function",
        field: "statistical_mechanics",
        equation: "Z = \\left(2\\cosh\\frac{J}{k_B T}\\right)^N",
        description: "Exact partition function for the 1D Ising model — a solvable model of phase transitions that illuminates cooperative phenomena in magnetism.",
        uses: ["var_partition_func", "energy", "const_kB", "temperature", "num_particles"],
        year: 1925, discoverer: "Ernst Ising"
    },
    {
        id: "equipartition_general",
        name: "Generalized Equipartition",
        field: "statistical_mechanics",
        equation: "\\left\\langle x_i \\frac{\\partial H}{\\partial x_j}\\right\\rangle = \\delta_{ij} k_B T",
        description: "Each quadratic degree of freedom contributes ½k_BT of energy on average — the most general form of the equipartition theorem.",
        uses: ["energy", "const_kB", "temperature"],
        year: 1876, discoverer: "Ludwig Boltzmann"
    },
    {
        id: "debye_heat_capacity",
        name: "Debye Heat Capacity",
        field: "statistical_mechanics",
        equation: "C_V = 9Nk_B\\left(\\frac{T}{\\Theta_D}\\right)^3 \\int_0^{\\Theta_D/T} \\frac{x^4 e^x}{(e^x-1)^2}\\,dx",
        description: "Correctly predicts the T³ behavior of solid heat capacity at low temperatures — a triumph of quantum statistical mechanics over the classical Dulong–Petit law.",
        uses: ["specific_heat", "num_particles", "const_kB", "temperature"],
        year: 1912, discoverer: "Peter Debye"
    },
    {
        id: "virial_expansion",
        name: "Virial Equation of State",
        field: "statistical_mechanics",
        equation: "\\frac{PV}{Nk_BT} = 1 + \\frac{B_2(T)}{V/N} + \\frac{B_3(T)}{(V/N)^2} + \\cdots",
        description: "Systematic correction to the ideal gas law using virial coefficients — each term accounts for interactions among increasingly many particles.",
        uses: ["pressure", "volume", "num_particles", "const_kB", "temperature", "compressibility"],
        year: 1901, discoverer: "Heike Kamerlingh Onnes"
    },

    // ─── Fluid Mechanics ────────────────────────────────────────────────────

    {
        id: "euler_fluid",
        name: "Euler's Equation (Fluid)",
        field: "fluid_mechanics",
        equation: "\\rho \\frac{D\\vec{v}}{Dt} = -\\nabla P + \\rho \\vec{g}",
        description: "Newton's second law for inviscid fluid flow — the Navier–Stokes equations without viscosity, governing ideal fluid dynamics.",
        uses: ["density", "velocity", "pressure", "force", "time"],
        year: 1757, discoverer: "Leonhard Euler"
    },
    {
        id: "kelvin_circulation",
        name: "Kelvin's Circulation Theorem",
        field: "fluid_mechanics",
        equation: "\\frac{D\\Gamma}{Dt} = 0",
        description: "In an ideal barotropic fluid with conservative body forces, the circulation around a material loop is constant — explains why vortices persist.",
        uses: ["velocity", "time"],
        year: 1869, discoverer: "Lord Kelvin"
    },
    {
        id: "mach_number",
        name: "Mach Number",
        field: "fluid_mechanics",
        equation: "\\text{Ma} = \\frac{v}{c_s}",
        description: "The ratio of flow speed to the local speed of sound — determines whether flow is subsonic, transonic, or supersonic, governing shock formation.",
        uses: ["velocity"],
        year: 1877, discoverer: "Ernst Mach"
    },
    {
        id: "vorticity_equation",
        name: "Vorticity Equation",
        field: "fluid_mechanics",
        equation: "\\frac{D\\vec{\\omega}}{Dt} = (\\vec{\\omega} \\cdot \\nabla)\\vec{v} + \\nu \\nabla^2 \\vec{\\omega}",
        description: "Evolution of vorticity in a viscous fluid — the first term is vortex stretching (3D turbulence), the second is diffusion. Fundamental to turbulence theory.",
        uses: ["angular_velocity", "velocity", "viscosity", "time"],
        year: 1858, discoverer: "Hermann von Helmholtz"
    },
    {
        id: "capillary_rise",
        name: "Capillary Rise (Jurin's Law)",
        field: "fluid_mechanics",
        equation: "h = \\frac{2\\gamma \\cos\\theta}{\\rho g r}",
        description: "Height that liquid rises in a narrow tube due to surface tension — balances surface tension force against the weight of the liquid column.",
        uses: ["height", "surface_tension", "angle", "density", "const_g", "radius"],
        year: 1718, discoverer: "James Jurin"
    },

    // ─── Optics & Waves ─────────────────────────────────────────────────────

    {
        id: "huygens_fresnel",
        name: "Huygens–Fresnel Principle",
        field: "optics_waves",
        equation: "U(P) = \\frac{i}{\\lambda} \\iint_\\Sigma U(Q) \\frac{e^{ikr}}{r}\\, dS",
        description: "Every point on a wavefront acts as a source of secondary spherical wavelets — the foundation of diffraction theory and Fourier optics.",
        uses: ["wavelength", "distance"],
        year: 1818, discoverer: "Augustin-Jean Fresnel"
    },
    {
        id: "abbe_diffraction_limit",
        name: "Abbe Diffraction Limit",
        field: "optics_waves",
        equation: "d = \\frac{\\lambda}{2n\\sin\\theta}",
        description: "The fundamental resolution limit of optical microscopy — no lens system can resolve features smaller than about half the wavelength of light.",
        uses: ["displacement", "wavelength", "refractive_index", "angle"],
        year: 1873, discoverer: "Ernst Abbe"
    },
    {
        id: "fresnel_reflection",
        name: "Fresnel Equations (Normal Incidence)",
        field: "optics_waves",
        equation: "R = \\left(\\frac{n_1 - n_2}{n_1 + n_2}\\right)^2",
        description: "Fraction of light intensity reflected at a boundary between two media at normal incidence — explains why glass reflects ~4% of incident light.",
        uses: ["refractive_index", "intensity"],
        year: 1823, discoverer: "Augustin-Jean Fresnel"
    },
    {
        id: "standing_wave",
        name: "Standing Wave Harmonics",
        field: "optics_waves",
        equation: "\\lambda_n = \\frac{2L}{n}",
        description: "The allowed wavelengths for standing waves on a string fixed at both ends — the physics behind musical instruments and quantum confinement.",
        uses: ["wavelength", "length"],
        year: 1746, discoverer: "Jean le Rond d'Alembert"
    },
    {
        id: "group_velocity",
        name: "Group Velocity",
        field: "optics_waves",
        equation: "v_g = \\frac{d\\omega}{dk}",
        description: "The velocity at which the envelope of a wave packet propagates — determines the speed at which energy and information travel in dispersive media.",
        uses: ["velocity", "angular_frequency", "wavelength"],
        year: 1877, discoverer: "Lord Rayleigh"
    },

    // ─── Nuclear Physics ────────────────────────────────────────────────────

    {
        id: "fermi_four_point",
        name: "Fermi's Interaction (β-decay)",
        field: "nuclear_physics",
        equation: "\\Gamma = \\frac{G_F^2 m_e^5 c^4}{2\\pi^3 \\hbar^7} f(Z, E_0)",
        description: "The decay rate for nuclear beta decay in Fermi's original four-fermion theory — the precursor to the full electroweak theory.",
        uses: ["const_me", "const_c", "const_hbar", "energy"],
        year: 1934, discoverer: "Enrico Fermi"
    },
    {
        id: "nuclear_shell_model",
        name: "Nuclear Magic Numbers",
        field: "nuclear_physics",
        equation: "E_{nlj} = E_{nl} - C \\vec{l}\\cdot\\vec{s}",
        description: "Nuclear energy levels with spin-orbit coupling — explains the magic numbers (2, 8, 20, 28, 50, 82, 126) where nuclei are exceptionally stable.",
        uses: ["energy", "angular_momentum", "spin_quantum", "orbital_quantum"],
        year: 1949, discoverer: "Maria Goeppert Mayer & J. Hans D. Jensen"
    },
    {
        id: "liquid_drop_energy",
        name: "Liquid Drop Model Energy",
        field: "nuclear_physics",
        equation: "E_B = a_V A - a_S A^{2/3} - a_C \\frac{Z(Z-1)}{A^{1/3}} - a_A \\frac{(A-2Z)^2}{A} + \\delta(A,Z)",
        description: "The semi-empirical mass formula — models the nucleus as a liquid drop with volume, surface, Coulomb, asymmetry, and pairing terms. Predicts binding energies to ~1%.",
        uses: ["energy", "mass_number", "atomic_number"],
        year: 1935, discoverer: "Carl Friedrich von Weizsäcker"
    },
    {
        id: "nuclear_fusion_rate",
        name: "Thermonuclear Reaction Rate",
        field: "nuclear_physics",
        equation: "r = n_1 n_2 \\langle \\sigma v \\rangle",
        description: "The rate of fusion reactions per unit volume — product of number densities and thermally-averaged cross section, determining whether a star can sustain fusion.",
        uses: ["number_density", "cross_section", "velocity"],
        year: 1938, discoverer: "Hans Bethe"
    },

    // ─── Astrophysics ───────────────────────────────────────────────────────

    {
        id: "virial_theorem_astro",
        name: "Virial Theorem (Astrophysics)",
        field: "astrophysics",
        equation: "2K + U = 0",
        description: "For a gravitationally bound system in equilibrium, the kinetic energy is minus half the potential energy — explains why stars heat up as they lose energy.",
        uses: ["energy"],
        year: 1870, discoverer: "Rudolf Clausius"
    },
    {
        id: "mass_luminosity",
        name: "Mass–Luminosity Relation",
        field: "astrophysics",
        equation: "L \\propto M^{3.5}",
        description: "Main-sequence stellar luminosity scales roughly as the 3.5 power of mass — massive stars burn vastly brighter but exhaust their fuel far faster.",
        uses: ["luminosity", "mass"],
        year: 1924, discoverer: "Arthur Eddington"
    },
    {
        id: "schwarzschild_isco",
        name: "ISCO Radius",
        field: "astrophysics",
        equation: "r_{\\text{ISCO}} = \\frac{6GM}{c^2} = 3r_s",
        description: "The innermost stable circular orbit around a non-rotating black hole — the inner edge of an accretion disk, where matter begins its final plunge.",
        uses: ["distance", "const_G", "mass", "const_c"],
        year: 1916, discoverer: "Karl Schwarzschild"
    },
    {
        id: "bondi_accretion",
        name: "Bondi Accretion Rate",
        field: "astrophysics",
        equation: "\\dot{M} = \\frac{4\\pi G^2 M^2 \\rho}{c_s^3}",
        description: "The rate at which a compact object accretes surrounding gas — balances gravitational pull against thermal pressure of the infalling material.",
        uses: ["mass", "const_G", "density", "velocity"],
        year: 1952, discoverer: "Hermann Bondi"
    },
    {
        id: "tully_fisher",
        name: "Tully–Fisher Relation",
        field: "astrophysics",
        equation: "L \\propto v_{\\text{rot}}^4",
        description: "Spiral galaxy luminosity correlates with rotation speed to the fourth power — a key cosmic distance indicator and evidence for dark matter.",
        uses: ["luminosity", "velocity"],
        year: 1977, discoverer: "R. Brent Tully & J. Richard Fisher"
    },

    // ─── Chemistry ──────────────────────────────────────────────────────────

    {
        id: "rate_law_general",
        name: "Rate Law",
        field: "chemistry",
        equation: "r = k [A]^m [B]^n",
        description: "The rate of a chemical reaction depends on reactant concentrations raised to their reaction orders — determined experimentally, not from stoichiometry.",
        uses: ["rate_constant", "concentration"],
        year: 1864, discoverer: "Cato Guldberg & Peter Waage"
    },
    {
        id: "integrated_first_order",
        name: "First-Order Integrated Rate Law",
        field: "chemistry",
        equation: "[A] = [A]_0 e^{-kt}",
        description: "Concentration decay in first-order kinetics — identical mathematical form to radioactive decay, reflecting a constant probability of reaction per unit time.",
        uses: ["concentration", "rate_constant", "time"],
        year: 1864, discoverer: "Augustus George Vernon Harcourt"
    },
    {
        id: "vant_hoff",
        name: "Van 't Hoff Equation",
        field: "chemistry",
        equation: "\\frac{d\\ln K}{dT} = \\frac{\\Delta H^\\circ}{RT^2}",
        description: "How the equilibrium constant changes with temperature — an exothermic reaction's equilibrium shifts toward reactants at higher temperature.",
        uses: ["equilibrium_constant", "temperature", "enthalpy", "const_R"],
        year: 1884, discoverer: "Jacobus Henricus van 't Hoff"
    },
    {
        id: "debye_huckel",
        name: "Debye–Hückel Limiting Law",
        field: "chemistry",
        equation: "\\log \\gamma_\\pm = -A|z_+ z_-|\\sqrt{I}",
        description: "Activity coefficient of ions in dilute solution decreases with ionic strength — accounts for long-range electrostatic interactions between ions.",
        uses: ["charge", "concentration"],
        year: 1923, discoverer: "Peter Debye & Erich Hückel"
    },
    {
        id: "osmotic_pressure",
        name: "Osmotic Pressure",
        field: "chemistry",
        equation: "\\Pi = MRT",
        description: "Osmotic pressure of a dilute solution is proportional to molar concentration and temperature — van 't Hoff's analogy to the ideal gas law for solutions.",
        uses: ["pressure", "concentration", "const_R", "temperature"],
        year: 1886, discoverer: "Jacobus Henricus van 't Hoff"
    },

    // ─── Information Theory ─────────────────────────────────────────────────

    {
        id: "channel_capacity_binary",
        name: "Binary Symmetric Channel",
        field: "information_theory",
        equation: "C = 1 - H(p) = 1 + p\\log_2 p + (1-p)\\log_2(1-p)",
        description: "Maximum reliable transmission rate through a noisy binary channel — where each bit can be flipped with probability p.",
        uses: ["channel_capacity", "probability"],
        year: 1948, discoverer: "Claude Shannon"
    },
    {
        id: "source_coding_theorem",
        name: "Source Coding Theorem",
        field: "information_theory",
        equation: "\\bar{L} \\geq H(X)",
        description: "No lossless compression can achieve an average code length less than the entropy of the source — the fundamental limit of data compression.",
        uses: ["information_entropy"],
        year: 1948, discoverer: "Claude Shannon"
    },
    {
        id: "fisher_information",
        name: "Fisher Information",
        field: "information_theory",
        equation: "I(\\theta) = E\\left[\\left(\\frac{\\partial \\ln f(X;\\theta)}{\\partial \\theta}\\right)^2\\right]",
        description: "Measures how much information an observable random variable carries about an unknown parameter — the precision limit of any unbiased estimator via the Cramér–Rao bound.",
        uses: ["probability"],
        year: 1925, discoverer: "Ronald A. Fisher"
    },
    {
        id: "holographic_entropy",
        name: "Holographic Entropy Bound",
        field: "information_theory",
        equation: "S \\leq \\frac{k_B c^3 A}{4 G \\hbar}",
        description: "Maximum entropy (information) in a region scales with the surface area, not volume — the profound insight that led to the holographic principle in quantum gravity.",
        uses: ["entropy", "const_kB", "const_c", "area", "const_G", "const_hbar"],
        year: 1995, discoverer: "Leonard Susskind"
    },

    // ─── Meta / Cross-Field ─────────────────────────────────────────────────

    {
        id: "dimensional_analysis",
        name: "Buckingham π Theorem",
        field: "meta",
        equation: "f(\\pi_1, \\pi_2, \\ldots, \\pi_{n-k}) = 0",
        description: "Any physically meaningful equation can be rewritten in terms of n−k dimensionless groups — the foundation of dimensional analysis and similitude in engineering.",
        uses: ["velocity", "length", "density", "viscosity"],
        year: 1914, discoverer: "Edgar Buckingham"
    },
    {
        id: "noether_theorem",
        name: "Noether's Theorem",
        field: "meta",
        equation: "\\frac{\\partial \\mathcal{L}}{\\partial q} = \\frac{d}{dt}\\frac{\\partial \\mathcal{L}}{\\partial \\dot{q}} \\;\\Rightarrow\\; Q = \\text{const}",
        description: "Every continuous symmetry of a physical system corresponds to a conserved quantity — time symmetry gives energy conservation, spatial symmetry gives momentum conservation.",
        uses: ["energy", "momentum", "angular_momentum"],
        year: 1918, discoverer: "Emmy Noether"
    },
    {
        id: "liouville_theorem",
        name: "Liouville's Theorem",
        field: "meta",
        equation: "\\frac{d\\rho}{dt} = \\frac{\\partial \\rho}{\\partial t} + \\{\\rho, H\\} = 0",
        description: "Phase-space density is conserved along trajectories in Hamiltonian systems — the bridge between classical mechanics and statistical mechanics.",
        uses: ["density", "time", "energy"],
        year: 1838, discoverer: "Joseph Liouville"
    },
    {
        id: "action_principle",
        name: "Principle of Least Action",
        field: "meta",
        equation: "\\delta S = \\delta \\int_{t_1}^{t_2} \\mathcal{L}\\,dt = 0",
        description: "Nature evolves along paths that make the action stationary — the deepest formulation of physics, from which all equations of motion can be derived.",
        uses: ["energy", "time"],
        year: 1744, discoverer: "Pierre Louis Maupertuis"
    },

    // ═════════════════════════════════════════════════════════════════════════
    // ORPHAN-FIX EQUATIONS — connecting previously isolated variables
    // ═════════════════════════════════════════════════════════════════════════

    // ─── Quantum Mechanics ──────────────────────────────────────────────────

    {
        id: "angular_momentum_z",
        name: "Orbital Angular Momentum z-Component",
        field: "quantum_mechanics",
        equation: "L_z = m_l \\hbar",
        description: "The z-component of orbital angular momentum is quantized in integer multiples of ℏ, labelled by the magnetic quantum number mₗ.",
        uses: ["angular_momentum", "magnetic_quantum", "const_hbar"],
        year: 1926, discoverer: "Erwin Schrödinger"
    },
    {
        id: "zeeman_energy",
        name: "Zeeman Effect Energy Splitting",
        field: "quantum_mechanics",
        equation: "\\Delta E = m_l \\mu_B B",
        description: "An external magnetic field splits atomic energy levels proportionally to the magnetic quantum number — each level fans into 2l+1 sub-levels.",
        uses: ["energy", "magnetic_quantum", "magnetic_moment", "magnetic_field"],
        year: 1896, discoverer: "Pieter Zeeman"
    },
    {
        id: "selection_rules_dipole",
        name: "Electric Dipole Selection Rules",
        field: "quantum_mechanics",
        equation: "\\Delta l = \\pm 1,\\quad \\Delta m_l = 0, \\pm 1",
        description: "Only transitions obeying these selection rules produce electric dipole radiation — they arise from the matrix element ⟨f|r|i⟩.",
        uses: ["orbital_quantum", "magnetic_quantum", "electric_dipole"],
        year: 1927, discoverer: "Paul Dirac"
    },

    // ─── Nuclear & Particle Physics ─────────────────────────────────────────

    {
        id: "impact_parameter_rutherford",
        name: "Rutherford Impact Parameter",
        field: "nuclear_physics",
        equation: "b = \\frac{Z_1 Z_2 e^2}{4E}\\cot\\!\\left(\\frac{\\theta}{2}\\right)",
        description: "Relates the impact parameter to the scattering angle in Coulomb scattering — larger b means smaller deflection.",
        uses: ["impact_parameter", "atomic_number", "const_e", "energy", "scattering_angle"],
        year: 1911, discoverer: "Ernest Rutherford"
    },
    {
        id: "cross_section_impact_param",
        name: "Differential Cross-Section (Impact Parameter)",
        field: "nuclear_physics",
        equation: "\\frac{d\\sigma}{d\\Omega} = \\frac{b}{\\sin\\theta}\\left|\\frac{db}{d\\theta}\\right|",
        description: "General relation between the differential cross-section and the impact parameter — valid for any central-force scattering.",
        uses: ["cross_section", "impact_parameter", "solid_angle", "scattering_angle"],
        year: 1911, discoverer: "Ernest Rutherford"
    },

    // ─── Astrophysics ───────────────────────────────────────────────────────

    {
        id: "keplerian_orbit",
        name: "Keplerian Orbit Equation",
        field: "astrophysics",
        equation: "r = \\frac{a(1-e^2)}{1 + e\\cos\\theta}",
        description: "The conic-section trajectory of a body in a gravitational field — ellipses (e<1), parabolas (e=1), and hyperbolas (e>1).",
        uses: ["distance", "semi_major_axis", "eccentricity", "angle"],
        year: 1609, discoverer: "Johannes Kepler"
    },
    {
        id: "orbital_eccentricity_energy",
        name: "Orbital Eccentricity from Energy",
        field: "astrophysics",
        equation: "e = \\sqrt{1 + \\frac{2EL^2}{m(GMm)^2}}",
        description: "Eccentricity determined by the orbit's total energy and angular momentum — bound orbits (E<0) have e<1.",
        uses: ["eccentricity", "energy", "angular_momentum", "mass", "const_G"],
        year: 1687, discoverer: "Isaac Newton"
    },
    {
        id: "vis_viva",
        name: "Vis-Viva Equation",
        field: "astrophysics",
        equation: "v^2 = GM\\left(\\frac{2}{r} - \\frac{1}{a}\\right)",
        description: "Relates orbital speed to position — the energy conservation law for Keplerian orbits. Gives escape velocity when a → ∞.",
        uses: ["velocity", "const_G", "mass", "distance", "semi_major_axis"],
        year: 1687, discoverer: "Isaac Newton"
    },
    {
        id: "hubble_redshift",
        name: "Hubble's Law (Redshift–Distance)",
        field: "astrophysics",
        equation: "v = H_0 d = cz",
        description: "For nearby galaxies, recession velocity is proportional to distance — the observational foundation of the expanding universe.",
        uses: ["velocity", "const_c", "distance", "redshift"],
        year: 1929, discoverer: "Edwin Hubble"
    },
    {
        id: "radiative_transfer",
        name: "Radiative Transfer Equation",
        field: "astrophysics",
        equation: "I(\\tau) = I_0\\, e^{-\\tau}",
        description: "Intensity falls exponentially with optical depth — the astrophysical form of the Beer-Lambert law for stellar atmospheres and interstellar media.",
        uses: ["intensity", "optical_depth"],
        year: 1906, discoverer: "Karl Schwarzschild"
    },
    {
        id: "optical_depth_definition",
        name: "Optical Depth Definition",
        field: "astrophysics",
        equation: "\\tau = \\int_0^L n\\,\\sigma\\, ds",
        description: "Optical depth accumulates along a line of sight as the product of number density and cross-section — τ≫1 means opaque.",
        uses: ["optical_depth", "number_density", "cross_section", "length"],
        year: 1906, discoverer: "Karl Schwarzschild"
    },

    // ─── Classical Mechanics ────────────────────────────────────────────────

    {
        id: "elastic_moduli_relation",
        name: "Elastic Moduli Relation",
        field: "classical_mechanics",
        equation: "G = \\frac{E}{2(1+\\nu)}",
        description: "Connects the shear modulus, Young's modulus, and Poisson's ratio — only two independent elastic constants exist for isotropic materials.",
        uses: ["shear_modulus", "youngs_modulus", "poisson_ratio"],
        year: 1850, discoverer: "Gabriel Lamé"
    },
    {
        id: "shear_stress_strain",
        name: "Shear Stress–Strain Relation",
        field: "classical_mechanics",
        equation: "\\tau = G\\gamma",
        description: "Shear stress is proportional to shear strain via the shear modulus — the torsional analogue of Hooke's law.",
        uses: ["stress", "shear_modulus", "strain"],
        year: 1784, discoverer: "Charles-Augustin de Coulomb"
    },
    {
        id: "bulk_youngs_poisson",
        name: "Bulk Modulus from Young's & Poisson's",
        field: "classical_mechanics",
        equation: "K = \\frac{E}{3(1-2\\nu)}",
        description: "Relates bulk modulus to Young's modulus and Poisson's ratio — diverges as ν → 0.5 (incompressible limit).",
        uses: ["bulk_modulus", "youngs_modulus", "poisson_ratio"],
        year: 1850, discoverer: "Gabriel Lamé"
    },

    // ─── Electromagnetism ───────────────────────────────────────────────────

    {
        id: "curie_law",
        name: "Curie's Law",
        field: "electromagnetism",
        equation: "M = \\frac{C\\,B}{T}",
        description: "Magnetization of a paramagnet is proportional to applied field and inversely proportional to temperature — thermal fluctuations fight alignment.",
        uses: ["magnetization", "magnetic_field", "temperature"],
        year: 1895, discoverer: "Pierre Curie"
    },
    {
        id: "magnetization_moment",
        name: "Magnetization–Dipole Moment Relation",
        field: "electromagnetism",
        equation: "\\vec{\\mu} = M\\,V",
        description: "Total magnetic dipole moment of a uniformly magnetized body equals magnetization times volume.",
        uses: ["magnetic_moment", "magnetization", "volume"],
        year: 1820, discoverer: "André-Marie Ampère"
    },
    {
        id: "drude_resistivity",
        name: "Drude Model Resistivity",
        field: "electromagnetism",
        equation: "\\rho = \\frac{m_e}{n\\,e^2\\,\\tau}",
        description: "Classical prediction of electrical resistivity from electron mass, density, and mean collision time — foundation of solid-state transport theory.",
        uses: ["resistivity_var", "const_me", "number_density", "const_e"],
        year: 1900, discoverer: "Paul Drude"
    },
    {
        id: "electric_dipole_radiation",
        name: "Electric Dipole Radiation Power",
        field: "electromagnetism",
        equation: "P = \\frac{\\ddot{p}^2}{6\\pi\\varepsilon_0 c^3}",
        description: "Power radiated by an oscillating electric dipole — the fundamental source term in antenna theory and atomic radiation.",
        uses: ["power", "electric_dipole", "const_eps0", "const_c"],
        year: 1897, discoverer: "Joseph Larmor"
    },

    // ─── Special Relativity ─────────────────────────────────────────────────

    {
        id: "spacetime_interval_proper",
        name: "Spacetime Interval (Proper Time)",
        field: "special_relativity",
        equation: "c^2 d\\tau^2 = c^2 dt^2 - dx^2 - dy^2 - dz^2",
        description: "The invariant spacetime interval expressed in terms of proper time — all observers agree on dτ even when they disagree on dt, dx, dy, dz.",
        uses: ["proper_time", "time", "const_c", "distance"],
        year: 1908, discoverer: "Hermann Minkowski"
    },

    // ─── Statistical Mechanics ──────────────────────────────────────────────

    {
        id: "landau_free_energy",
        name: "Landau Free Energy",
        field: "statistical_mechanics",
        equation: "F = F_0 + \\alpha(T-T_c)\\,\\phi^2 + \\beta\\,\\phi^4",
        description: "Phenomenological expansion of free energy near a phase transition — the sign of α(T−Tc) determines whether the ordered phase (φ≠0) is stable.",
        uses: ["energy", "temperature", "order_parameter", "critical_exponent"],
        year: 1937, discoverer: "Lev Landau"
    },
    {
        id: "order_parameter_scaling",
        name: "Order Parameter Critical Scaling",
        field: "statistical_mechanics",
        equation: "\\phi \\propto (T_c - T)^{\\beta}",
        description: "Near a continuous phase transition the order parameter vanishes as a power law — the exponent β is universal, depending only on dimensionality and symmetry.",
        uses: ["order_parameter", "temperature", "critical_exponent"],
        year: 1937, discoverer: "Lev Landau"
    },
    {
        id: "correlation_length_divergence",
        name: "Correlation Length Divergence",
        field: "statistical_mechanics",
        equation: "\\xi = \\xi_0\\,|T - T_c|^{-\\nu}",
        description: "The correlation length diverges at the critical point — fluctuations become correlated over all length scales, producing critical opalescence.",
        uses: ["correlation_length", "temperature"],
        year: 1944, discoverer: "Lars Onsager"
    },
    {
        id: "ornstein_zernike_correlation",
        name: "Ornstein-Zernike Correlation Function",
        field: "statistical_mechanics",
        equation: "G(r) \\propto \\frac{e^{-r/\\xi}}{r^{d-2+\\eta}}",
        description: "Spatial decay of correlations near a phase transition — exponential decay with range ξ, modulated by a power-law prefactor with anomalous dimension η.",
        uses: ["correlation_length", "distance"],
        year: 1914, discoverer: "Leonard Ornstein"
    },
    {
        id: "mean_free_path_kinetic",
        name: "Mean Free Path (Kinetic Theory)",
        field: "statistical_mechanics",
        equation: "\\lambda = \\frac{1}{\\sqrt{2}\\,n\\,\\sigma}",
        description: "Average distance a particle travels between collisions — inversely proportional to number density and cross-section.",
        uses: ["mean_free_path", "number_density", "cross_section"],
        year: 1858, discoverer: "Rudolf Clausius"
    },

    // ─── Connectivity Boost Equations ────────────────────────────────────────

    {
        id: "displacement_field",
        name: "Electric Displacement Field",
        field: "electromagnetism",
        equation: "\\vec{D} = \\varepsilon_0 \\vec{E} + \\vec{P} = \\varepsilon_0(1+\\chi_e)\\vec{E}",
        description: "The displacement field accounts for both free and bound charges — in linear media it separates the material response (P) from the vacuum contribution.",
        uses: ["electric_field", "const_eps0", "polarization", "electric_susceptibility"],
        year: 1864, discoverer: "James Clerk Maxwell"
    },
    {
        id: "compressibility_definition",
        name: "Isothermal Compressibility",
        field: "thermodynamics",
        equation: "\\kappa_T = -\\frac{1}{V}\\left(\\frac{\\partial V}{\\partial P}\\right)_T",
        description: "Fractional volume change per unit pressure at constant temperature — the reciprocal of the bulk modulus.",
        uses: ["compressibility", "volume", "pressure", "temperature"],
        year: 1834, discoverer: "Benoît Clapeyron"
    },
    {
        id: "friis_formula",
        name: "Friis Transmission Formula",
        field: "information_theory",
        equation: "\\frac{S}{N} = \\frac{P_t G_t G_r \\lambda^2}{(4\\pi d)^2 k_B T B}",
        description: "Signal-to-noise ratio for a radio link — combines antenna gains, path loss, and thermal noise floor.",
        uses: ["signal_noise_ratio", "power", "wavelength", "distance", "const_kB", "temperature", "bandwidth"],
        year: 1946, discoverer: "Harald T. Friis"
    },
];

// ─── Status Categories ───────────────────────────────────────────────────────
// Each equation has a STATUS — what it fundamentally IS in physics:
//   postulate     — Fundamental law; not derived, only tested experimentally
//   derived       — Proven mathematical consequence of postulates
//   solution      — Exact closed-form solution to a fundamental equation
//   approximation — Deliberately simplified; valid under restricted conditions
//   empirical     — Observed experimentally; lacks first-principles derivation
//   definition    — Naming convention or mathematical identity; true by definition

const STATUS_INFO = {
    postulate:     { label: "Postulate",     badge: "✦", color: "#FFD700" },
    derived:       { label: "Derived",       badge: "⊢", color: "#e0e4ec" },
    solution:      { label: "Solution",      badge: "◉", color: "#00e676" },
    approximation: { label: "Approximation", badge: "≈", color: "#FF8C00" },
    empirical:     { label: "Empirical",     badge: "⚗", color: "#FF6B9D" },
    definition:    { label: "Definition",    badge: "≡", color: "#8892a4" },
};

// ═══════════════════════════════════════════════════════════════════════════════
// EQUATION METADATA — Clean Taxonomy of Physical Equations
// ═══════════════════════════════════════════════════════════════════════════════
//
//   status:          one of: postulate | derived | solution | approximation | empirical | definition
//   conditions:      when/where the equation is valid (shown in detail panel)
//   supersededBy:    ID of the more general equation (Generalizes relationship)
//   equivalentTo:    array of IDs — same physics, different formulation
//   incompatibleWith: array of IDs — mutually exclusive assumptions
//   derivesFrom:     array of { eq, assuming } — logical derivation chain
//   componentOf:     string — system this equation belongs to (e.g. "maxwell_equations")
//   statusNote:      historical context or disambiguation
//   forms/formNotes: alternative mathematical representations
//
// Unlisted equations default to status: "derived".
// ═══════════════════════════════════════════════════════════════════════════════

const EQUATION_META = {

    // ─── Classical Mechanics ────────────────────────────────────

    "newton_second_law": {
        status: "approximation",
        conditions: "Constant mass, v \ll c",
        supersededBy: "newton_second_momentum",
        equivalentTo: ["euler_lagrange", "hamiltons_equations"],
        incompatibleWith: ["heisenberg_xp"],
        forms: { differential: "\\vec{F} = \\frac{d\\vec{p}}{dt}" },
        formNotes: { differential: "Handles variable mass (rockets, relativistic particles)" },
    },

    "kinetic_energy": {
        status: "approximation",
        conditions: "v \ll c",
        supersededBy: "relativistic_ke",
        forms: { integral: "E_k = \\int_0^v mv' \\, dv'" },
        formNotes: { integral: "Derivable from integrating F·ds with constant mass" },
    },

    "gravitational_pe_surface": {
        status: "approximation",
        conditions: "h \ll R_{\oplus} (near Earth's surface)",
        supersededBy: "gravitational_pe_general",
    },

    "newton_gravitation": {
        status: "approximation",
        conditions: "Weak fields, v \ll c, flat spacetime",
        statusNote: "Was a postulate (1687); now understood as the weak-field limit of general relativity",
        supersededBy: "einstein_field_equations",
        incompatibleWith: ["gravitational_time_dilation"],
        forms: { vector: "\\vec{F} = -\\frac{GMm}{r^2}\\hat{r}" },
    },

    "gravitational_pe_general": {
        status: "derived",
        conditions: "Weak gravitational fields",
        supersededBy: "einstein_field_equations",
        derivesFrom: [{ eq: "newton_gravitation", assuming: "Conservative force field" }],
    },

    "momentum_def": {
        status: "definition",
        conditions: "Classical mechanics",
        supersededBy: "relativistic_momentum",
    },

    "centripetal_force": {
        status: "derived",
        conditions: "Uniform circular motion",
        derivesFrom: [{ eq: "newton_second_law", assuming: "Circular path, constant speed" }],
    },

    "angular_momentum_def": {
        status: "definition",
        conditions: "Rigid body rotation",
        derivesFrom: [{ eq: "newton_second_momentum", assuming: "rotational analog" }],
    },

    "torque_def": {
        status: "derived",
        conditions: "Rigid body rotation",
        statusNote: "Rotational analog of Newton's second law",
        derivesFrom: [{ eq: "newton_second_momentum", assuming: "rotational form" }],
    },

    "rotational_ke": {
        status: "derived",
        conditions: "Rigid body rotation",
        derivesFrom: [{ eq: "kinetic_energy", assuming: "rigid body rotation" }],
    },

    "hookes_law": {
        status: "empirical",
        conditions: "Small deformations, elastic regime",
        statusNote: "Empirical observation; breaks down for large deformations and nonlinear materials",
    },

    "shm_spring_period": {
        status: "derived",
        conditions: "Linear restoring force (Hooke's law regime)",
        derivesFrom: [{ eq: "hookes_law", assuming: "SHM in linear elastic regime" }],
    },

    "shm_pendulum_period": {
        status: "approximation",
        conditions: "\theta \ll 1 \text{ rad (small angle)}",
        derivesFrom: [{ eq: "newton_second_law", assuming: "small angle sin θ ≈ θ" }],
    },

    "escape_velocity": {
        status: "derived",
        conditions: "Non-relativistic, spherically symmetric mass",
        derivesFrom: [{ eq: "gravitational_pe_general", assuming: "KE = |PE| at escape" }],
    },

    "kepler_third_law": {
        status: "derived",
        conditions: "Two-body, M \gg m, circular orbits",
        statusNote: "Originally empirical (1619); later derived from Newtonian gravity",
        supersededBy: "newton_gravitation",
        derivesFrom: [{ eq: "newton_gravitation", assuming: "Circular orbit, M ≫ m" }],
    },

    "work_def": {
        status: "definition",
        conditions: "Constant force, straight path",
        derivesFrom: [{ eq: "newton_second_law", assuming: "integrated over displacement" }],
        forms: { integral: "W = \\int \\vec{F} \\cdot d\\vec{s}" },
        formNotes: { integral: "General form for varying force along an arbitrary path" },
    },

    "power_mechanical": {
        status: "definition",
        derivesFrom: [{ eq: "work_def", assuming: "time derivative" }],
        forms: { differential: "P = \\frac{dW}{dt}" },
        formNotes: { differential: "Instantaneous power as the time derivative of work" },
    },

    "orbital_velocity": {
        status: "derived",
        conditions: "Circular orbit, Newtonian gravity",
        derivesFrom: [{ eq: "newton_gravitation", assuming: "Circular orbit equilibrium" }],
    },

    "weight": {
        status: "definition",
        conditions: "Near Earth's surface",
        derivesFrom: [{ eq: "newton_second_law", assuming: "a = g near Earth surface" }],
    },

    "angular_momentum_particle": {
        status: "definition",
        conditions: "Particle in circular motion",
        derivesFrom: [{ eq: "angular_momentum_def", assuming: "point particle" }],
    },

    "torque_cross": {
        status: "definition",
        derivesFrom: [{ eq: "torque_def", assuming: "cross product form" }],
    },

    "angular_kinematics": {
        status: "derived",
        conditions: "Constant angular acceleration",
        derivesFrom: [{ eq: "newton_second_law", assuming: "constant angular acceleration" }],
    },

    "sphere_volume": {
        status: "definition",
        statusNote: "Mathematical identity",
        componentOf: "geometric_formulas",
    },

    "sphere_surface": {
        status: "definition",
        statusNote: "Mathematical identity",
        componentOf: "geometric_formulas",
    },

    "galilean_velocity_addition": {
        status: "approximation",
        conditions: "v_1, v_2 \ll c",
        supersededBy: "relativistic_velocity_addition",
        incompatibleWith: ["lorentz_transformation"],
    },

    "newton_second_momentum": {
        status: "postulate",
        conditions: "All classical & relativistic mechanics",
        equivalentTo: ["euler_lagrange", "hamiltons_equations"],
    },

    "euler_lagrange": {
        status: "postulate",
        conditions: "All classical mechanics (holonomic constraints)",
        statusNote: "Derived from the principle of stationary action, which is itself postulated",
        equivalentTo: ["newton_second_law", "hamiltons_equations"],
    },

    "hamiltons_equations": {
        status: "postulate",
        conditions: "All Hamiltonian mechanics",
        equivalentTo: ["newton_second_law", "euler_lagrange"],
        forms: { differential: "\\dot{q}_i = \\frac{\\partial H}{\\partial p_i}, \\quad \\dot{p}_i = -\\frac{\\partial H}{\\partial q_i}" },
    },

    "virial_theorem": {
        status: "derived",
        conditions: "Time-averaged stable systems (bounded orbits)",
        derivesFrom: [{ eq: "newton_second_law", assuming: "time-averaged bound system" }],
    },

    "elastic_pe": {
        status: "derived",
        conditions: "Hooke's law regime (linear elasticity)",
        derivesFrom: [{ eq: "hookes_law", assuming: "Work done compressing a spring" }],
    },

    "impulse_momentum": {
        status: "derived",
        conditions: "All mechanics",
        derivesFrom: [{ eq: "newton_second_momentum", assuming: "Integration over time interval" }],
        forms: { integral: "\\vec{J} = \\int_{t_1}^{t_2} \\vec{F}\\,dt = \\Delta \\vec{p}" },
    },

    "mass_density": {
        status: "definition",
        derivesFrom: [{ eq: "newton_second_law", assuming: "ρ = m/V; intensive property of matter" }],
    },


    // ─── Electromagnetism ───────────────────────────────────────

    "coulombs_law": {
        status: "derived",
        conditions: "Static point charges in vacuum",
        derivesFrom: [{ eq: "gauss_law", assuming: "Spherical symmetry, point charge" }],
        forms: { vector: "\\vec{F} = k_e \\frac{q_1 q_2}{r^2}\\hat{r}" },
    },

    "electric_field_point": {
        status: "derived",
        conditions: "r \gg 0,\; \text{point charge}",
        derivesFrom: [{ eq: "coulombs_law", assuming: "field per unit test charge" }],
    },

    "gauss_law": {
        status: "postulate",
        conditions: "All electric field configurations",
        equivalentTo: ["ampere_maxwell", "gauss_magnetism", "faradays_law"],
        componentOf: "maxwell_equations",
        forms: { differential: "\\nabla \\cdot \\vec{E} = \\frac{\\rho}{\\varepsilon_0}" },
        formNotes: { differential: "Local (differential) form of Maxwell's first equation" },
    },

    "electric_potential_point": {
        status: "derived",
        conditions: "r \gg 0,\; \text{point charge}",
        derivesFrom: [{ eq: "coulombs_law", assuming: "integrated over path" }],
    },

    "coulomb_pe": {
        status: "derived",
        derivesFrom: [{ eq: "coulombs_law", assuming: "work to assemble charges" }],
    },

    "parallel_plate_capacitance": {
        status: "derived",
        conditions: "\text{parallel plates},\; A \gg d^2",
        derivesFrom: [{ eq: "gauss_law", assuming: "uniform field between plates" }],
    },

    "capacitor_energy": {
        status: "derived",
        derivesFrom: [{ eq: "electric_potential_point", assuming: "energy stored in field" }],
    },

    "ohms_law": {
        status: "empirical",
        conditions: "Ohmic materials, constant temperature",
        statusNote: "Phenomenological; breaks down in superconductors, semiconductors, plasmas",
    },

    "electric_power": {
        status: "definition",
        derivesFrom: [{ eq: "ohms_law", assuming: "P = IV from energy conservation" }],
    },

    "biot_savart": {
        status: "derived",
        conditions: "Steady currents (magnetostatics)",
        derivesFrom: [{ eq: "ampere_maxwell", assuming: "Steady current, no displacement current" }],
        forms: { integral: "\\vec{B} = \\frac{\\mu_0}{4\\pi} \\int \\frac{I \\, d\\vec{\\ell} \\times \\hat{r}}{r^2}" },
        formNotes: { integral: "General Biot-Savart for arbitrary current paths" },
    },

    "faradays_law": {
        status: "postulate",
        conditions: "All electromagnetic induction",
        componentOf: "maxwell_equations",
        forms: { integral: "\\oint \\vec{E} \\cdot d\\vec{\\ell} = -\\frac{d\\Phi_B}{dt}", differential: "\\nabla \\times \\vec{E} = -\\frac{\\partial \\vec{B}}{\\partial t}" },
        formNotes: { integral: "Integral form: circulation of E around a loop", differential: "Maxwell's third equation in differential form" },
    },

    "lorentz_force": {
        status: "postulate",
        conditions: "All charged particle motion in EM fields",
        statusNote: "Fundamental force law of electromagnetism; relativistically exact",
        incompatibleWith: ["newton_second_law"],
    },

    "em_wave_speed": {
        status: "derived",
        conditions: "Follows from Maxwell's equations",
        derivesFrom: [{ eq: "ampere_maxwell", assuming: "All four Maxwell equations in vacuum" }],
    },

    "magnetic_force_wire": {
        status: "derived",
        derivesFrom: [{ eq: "lorentz_force", assuming: "current = moving charges" }],
    },

    "inductor_energy": {
        status: "derived",
        derivesFrom: [{ eq: "faradays_law", assuming: "energy stored in magnetic field" }],
    },

    "resistivity": {
        status: "derived",
        conditions: "Uniform cross-section conductor",
        derivesFrom: [{ eq: "ohms_law", assuming: "uniform conductor geometry" }],
    },

    "joule_heating": {
        status: "derived",
        derivesFrom: [{ eq: "ohms_law", assuming: "P = IV with V = IR" }],
    },

    "inductor_emf": {
        status: "derived",
        derivesFrom: [{ eq: "faradays_law", assuming: "Self-inductance" }],
    },

    "magnetic_flux_def": {
        status: "definition",
        derivesFrom: [{ eq: "faradays_law", assuming: "Φ_B = ∫B·dA; defined for Faraday's law" }],
    },

    "ampere_maxwell": {
        status: "postulate",
        conditions: "All electromagnetic phenomena",
        componentOf: "maxwell_equations",
        forms: { differential: "\\nabla \\times \\vec{B} = \\mu_0\\vec{J} + \\mu_0\\varepsilon_0\\frac{\\partial \\vec{E}}{\\partial t}", integral: "\\oint \\vec{B} \\cdot d\\vec{\\ell} = \\mu_0 I_{enc} + \\mu_0\\varepsilon_0\\frac{d\\Phi_E}{dt}" },
        formNotes: { differential: "Maxwell's 4th equation — local form", integral: "Integral form including displacement current" },
    },

    "gauss_magnetism": {
        status: "postulate",
        conditions: "No magnetic monopoles",
        componentOf: "maxwell_equations",
        forms: { integral: "\\oint \\vec{B} \\cdot d\\vec{A} = 0" },
        formNotes: { integral: "Integral form: net magnetic flux through any closed surface is zero" },
    },

    "poynting_vector": {
        status: "derived",
        conditions: "All electromagnetic field configurations",
        derivesFrom: [{ eq: "ampere_maxwell", assuming: "Energy conservation + Maxwell's equations" }],
    },

    "em_wave_equation": {
        status: "derived",
        conditions: "Source-free vacuum",
        derivesFrom: [{ eq: "ampere_maxwell", assuming: "Vacuum, no sources" }],
    },

    "solenoid_field": {
        status: "approximation",
        conditions: "Ideal solenoid (infinite length, uniform winding)",
        derivesFrom: [{ eq: "ampere_maxwell", assuming: "Infinite solenoid symmetry" }],
    },

    "lc_resonance": {
        status: "derived",
        conditions: "Ideal LC circuit (no resistance)",
        derivesFrom: [{ eq: "faradays_law", assuming: "LC circuit with no resistance" }],
    },

    "cyclotron_frequency": {
        status: "derived",
        conditions: "Non-relativistic, uniform B field",
        derivesFrom: [{ eq: "lorentz_force", assuming: "Uniform B, circular motion" }],
    },

    "dipole_field": {
        status: "approximation",
        conditions: "Far field (r \gg d), point dipole",
        supersededBy: "coulombs_law",
    },


    // ─── Thermodynamics ─────────────────────────────────────────

    "ideal_gas_law": {
        status: "approximation",
        conditions: "Low pressure, high T (no intermolecular forces)",
        supersededBy: "van_der_waals",
        incompatibleWith: ["van_der_waals"],
    },

    "kinetic_theory_pressure": {
        status: "derived",
        conditions: "Ideal gas, isotropic velocities",
        derivesFrom: [{ eq: "newton_second_law", assuming: "molecular collisions with wall" }],
    },

    "first_law_thermo": {
        status: "postulate",
        conditions: "All thermodynamic systems",
        componentOf: "laws_of_thermodynamics",
    },

    "entropy_clausius": {
        status: "definition",
        conditions: "Reversible processes",
        statusNote: "Thermodynamic definition of entropy; superseded by the statistical definition",
        supersededBy: "boltzmann_entropy",
    },

    "boltzmann_entropy": {
        status: "postulate",
        conditions: "All thermodynamic systems",
        statusNote: "Bridges thermodynamics to statistical mechanics; engraved on Boltzmann's tombstone",
        equivalentTo: ["entropy_clausius"],
    },

    "helmholtz_free_energy": {
        status: "definition",
        conditions: "Constant T, V processes",
        componentOf: "thermodynamic_potentials",
    },

    "gibbs_free_energy": {
        status: "definition",
        conditions: "Constant T, P processes",
        derivesFrom: [{ eq: "first_law_thermo", assuming: "Legendre transform at constant T, P" }],
        componentOf: "thermodynamic_potentials",
    },

    "enthalpy_def": {
        status: "definition",
        conditions: "Constant pressure processes",
        componentOf: "thermodynamic_potentials",
    },

    "heat_capacity": {
        status: "derived",
        conditions: "Constant specific heat over ΔT",
        derivesFrom: [{ eq: "first_law_thermo", assuming: "constant volume or pressure process" }],
    },

    "carnot_efficiency": {
        status: "derived",
        conditions: "All heat engines (theoretical maximum)",
        derivesFrom: [{ eq: "clausius_inequality", assuming: "Reversible cycle between two reservoirs" }],
    },

    "stefan_boltzmann": {
        status: "derived",
        conditions: "All blackbody radiators",
        derivesFrom: [{ eq: "planck_law", assuming: "Integration over all frequencies" }],
    },

    "wiens_law": {
        status: "derived",
        conditions: "Blackbody emission peak",
        statusNote: "Approximate — the exact relation requires solving a transcendental equation",
        derivesFrom: [{ eq: "planck_law", assuming: "Maximizing spectral radiance" }],
    },

    "planck_law": {
        status: "postulate",
        conditions: "All frequencies & temperatures",
        statusNote: "Resolved the ultraviolet catastrophe; launched quantum theory (1900)",
        incompatibleWith: ["rayleigh_jeans"],
    },

    "maxwell_boltzmann": {
        status: "derived",
        conditions: "Classical ideal gas, distinguishable particles",
        supersededBy: "fermi_dirac",
        incompatibleWith: ["fermi_dirac", "bose_einstein"],
    },

    "equipartition": {
        status: "derived",
        conditions: "Classical regime (kT ≫ quantum energy spacing)",
        derivesFrom: [{ eq: "boltzmann_distribution", assuming: "classical limit, quadratic degrees of freedom" }],
    },

    "fourier_law": {
        status: "empirical",
        conditions: "Conductive heat transfer",
        statusNote: "Phenomenological heat conduction; fails at nanoscale and ballistic regimes",
        forms: { differential: "\\vec{q} = -\\kappa \\nabla T" },
    },

    "clausius_clapeyron": {
        status: "derived",
        conditions: "Phase equilibrium",
        derivesFrom: [{ eq: "gibbs_free_energy", assuming: "Two-phase equilibrium at constant P, T" }],
    },

    "heat_engine_efficiency": {
        status: "definition",
        derivesFrom: [{ eq: "first_law_thermo", assuming: "cyclic engine process" }],
    },

    "dulong_petit": {
        status: "approximation",
        conditions: "T \gg \Theta_D (high temperature limit)",
        derivesFrom: [{ eq: "equipartition", assuming: "3 vibrational modes per atom, classical limit" }],
    },

    "fourier_steady": {
        status: "derived",
        conditions: "Steady-state, uniform slab",
        derivesFrom: [{ eq: "fourier_law", assuming: "Steady state, planar geometry" }],
    },

    "internal_energy_ideal": {
        status: "derived",
        conditions: "Ideal gas",
        derivesFrom: [{ eq: "equipartition", assuming: "ideal gas, no intermolecular forces" }],
    },

    "chemical_potential_def": {
        status: "definition",
        componentOf: "thermodynamic_potentials",
    },

    "van_der_waals": {
        status: "empirical",
        conditions: "Real gases (semi-empirical corrections)",
        statusNote: "Phenomenological correction for molecular volume and intermolecular forces",
    },

    "clausius_inequality": {
        status: "postulate",
        conditions: "All thermodynamic cycles",
        statusNote: "Mathematical statement of the Second Law of Thermodynamics",
        componentOf: "laws_of_thermodynamics",
    },

    "maxwell_relation_tv": {
        status: "derived",
        conditions: "All thermodynamic potentials",
        derivesFrom: [{ eq: "helmholtz_free_energy", assuming: "exactness of dF" }],
    },

    "adiabatic_process": {
        status: "derived",
        conditions: "Reversible adiabatic, ideal gas",
        derivesFrom: [{ eq: "first_law_thermo", assuming: "Q = 0, ideal gas" }],
    },

    "third_law_thermo": {
        status: "postulate",
        conditions: "All crystalline systems as T → 0",
        componentOf: "laws_of_thermodynamics",
    },

    "ideal_gas_kinetic": {
        status: "derived",
        conditions: "Classical ideal gas",
        statusNote: "Temperature IS average kinetic energy — deepest bridge between mechanics and thermodynamics",
        derivesFrom: [{ eq: "kinetic_theory_pressure", assuming: "combined with ideal gas law" }],
    },


    // ─── Quantum Mechanics ──────────────────────────────────────

    "planck_einstein": {
        status: "postulate",
        conditions: "All photons",
        statusNote: "Einstein's 1905 light-quantum hypothesis; generalized Planck's oscillator quantization",
    },

    "de_broglie": {
        status: "postulate",
        conditions: "All matter (non-relativistic)",
        statusNote: "de Broglie's 1924 hypothesis; confirmed by Davisson-Germer experiment (1927)",
        derivesFrom: [{ eq: "planck_einstein", assuming: "wave-particle duality extended to matter" }],
    },

    "photon_momentum": {
        status: "derived",
        derivesFrom: [
            { eq: "planck_einstein", assuming: "combined with E = pc for massless particle" },
            { eq: "energy_momentum_relation", assuming: "m = 0" }
        ],
    },

    "heisenberg_xp": {
        status: "postulate",
        conditions: "All quantum systems",
        derivesFrom: [{ eq: "commutation_xp", assuming: "Robertson uncertainty relation" }],
    },

    "heisenberg_et": {
        status: "derived",
        conditions: "All quantum systems",
        derivesFrom: [{ eq: "heisenberg_xp", assuming: "energy-time analog via Fourier analysis" }],
    },

    "schrodinger_td": {
        status: "postulate",
        conditions: "Non-relativistic quantum mechanics",
        equivalentTo: ["schrodinger_ti"],
        incompatibleWith: ["lorentz_transformation"],
        forms: { differential: "-\\frac{\\hbar^2}{2m}\\nabla^2\\psi + V\\psi = i\\hbar\\frac{\\partial\\psi}{\\partial t}" },
        formNotes: { differential: "Explicit form with kinetic and potential energy operators" },
    },

    "schrodinger_ti": {
        status: "derived",
        conditions: "Stationary states (non-relativistic)",
        equivalentTo: ["schrodinger_td"],
        derivesFrom: [{ eq: "schrodinger_td", assuming: "Time-independent potential, separation of variables" }],
        forms: { differential: "-\\frac{\\hbar^2}{2m}\\nabla^2\\psi + V\\psi = E\\psi" },
    },

    "photoelectric": {
        status: "derived",
        conditions: "Metal surfaces, monochromatic photons",
        derivesFrom: [{ eq: "planck_einstein", assuming: "Energy conservation at metal surface" }],
    },

    "compton_scattering": {
        status: "derived",
        conditions: "Photon-electron scattering (free electron)",
        derivesFrom: [{ eq: "energy_momentum_relation", assuming: "Photon-electron elastic collision" }],
    },

    "hydrogen_energy": {
        status: "solution",
        conditions: "Non-relativistic, single electron, 1/r potential",
        supersededBy: "schrodinger_ti",
        derivesFrom: [{ eq: "schrodinger_ti", assuming: "Coulomb potential, spherical coordinates" }],
    },

    "bohr_radius": {
        status: "derived",
        conditions: "Ground-state hydrogen",
        derivesFrom: [{ eq: "schrodinger_ti", assuming: "Coulomb potential, ground state" }],
    },

    "fine_structure_constant": {
        status: "definition",
        statusNote: "Dimensionless coupling constant of QED; α ≈ 1/137",
        derivesFrom: [{ eq: "coulombs_law", assuming: "α = k_e e²/(ℏc); ratio of EM to quantum scales" }],
    },

    "rydberg_formula": {
        status: "derived",
        conditions: "Hydrogen-like atoms",
        statusNote: "Originally empirical (1888); later derived from the Bohr model and quantum mechanics",
        derivesFrom: [{ eq: "hydrogen_energy", assuming: "Transition between energy levels" }],
    },

    "thermal_de_broglie": {
        status: "derived",
        conditions: "Particles at temperature T",
        derivesFrom: [{ eq: "de_broglie", assuming: "thermal average momentum p = √(2πmk_BT)" }],
    },

    "threshold_frequency": {
        status: "derived",
        derivesFrom: [{ eq: "planck_einstein", assuming: "Minimum energy = work function" }],
    },

    "proton_compton": {
        status: "derived",
        derivesFrom: [{ eq: "de_broglie", assuming: "applied to proton rest mass" }],
    },

    "born_rule": {
        status: "postulate",
        conditions: "All quantum measurements",
        statusNote: "Connects wavefunction to measurement probabilities; still debated in foundations",
    },

    "commutation_xp": {
        status: "postulate",
        conditions: "All non-relativistic quantum mechanics",
        statusNote: "Foundational postulate of quantum mechanics; encodes non-commutativity",
    },

    "particle_in_box": {
        status: "solution",
        conditions: "Infinite square well potential",
        derivesFrom: [{ eq: "schrodinger_ti", assuming: "Infinite potential walls" }],
    },

    "qm_harmonic_oscillator": {
        status: "solution",
        conditions: "Quadratic potential V = ½mω²x²",
        statusNote: "Includes zero-point energy ½ℏω — vacuum is never truly empty",
        derivesFrom: [{ eq: "schrodinger_ti", assuming: "Harmonic potential" }],
    },

    "dirac_equation": {
        status: "postulate",
        conditions: "Relativistic spin-½ particles",
        statusNote: "Predicted the positron and intrinsic spin/magnetic moment of the electron",
        incompatibleWith: ["schrodinger_td"],
    },

    "fermis_golden_rule": {
        status: "derived",
        conditions: "First-order perturbation theory",
        derivesFrom: [{ eq: "schrodinger_td", assuming: "Weak perturbation, first order" }],
    },

    "tunnel_transmission": {
        status: "approximation",
        conditions: "Rectangular barrier, WKB approximation",
        supersededBy: "schrodinger_td",
    },

    "ehrenfest_theorem": {
        status: "derived",
        conditions: "All quantum mechanics",
        statusNote: "The bridge between quantum and classical physics",
        derivesFrom: [{ eq: "schrodinger_td", assuming: "Expectation values of observables" }],
    },

    "klein_gordon": {
        status: "postulate",
        conditions: "Relativistic spin-0 particles",
        statusNote: "First attempt at relativistic QM; limited to spin-0",
        supersededBy: "dirac_equation",
    },

    "spin_angular_momentum": {
        status: "derived",
        conditions: "All quantum particles with spin",
        derivesFrom: [{ eq: "dirac_equation", assuming: "intrinsic angular momentum from Dirac theory" }],
    },

    "magnetic_moment_orbital": {
        status: "derived",
        conditions: "Non-relativistic, no spin-orbit coupling",
        derivesFrom: [{ eq: "angular_momentum_def", assuming: "charged particle in orbit" }],
    },


    // ─── Special Relativity ─────────────────────────────────────

    "mass_energy": {
        status: "derived",
        conditions: "Rest frame only",
        statusNote: "The most famous equation in physics",
        supersededBy: "energy_momentum_relation",
        derivesFrom: [{ eq: "lorentz_transformation", assuming: "Rest frame (p = 0)" }],
    },

    "energy_momentum_relation": {
        status: "derived",
        conditions: "All reference frames",
        incompatibleWith: ["kinetic_energy"],
    },

    "lorentz_factor_def": {
        status: "definition",
        derivesFrom: [{ eq: "lorentz_transformation", assuming: "defined from transformation" }],
    },

    "time_dilation": {
        status: "derived",
        conditions: "Inertial frames",
        derivesFrom: [{ eq: "lorentz_transformation", assuming: "Co-located events in moving frame" }],
    },

    "length_contraction": {
        status: "derived",
        conditions: "Inertial frames",
        derivesFrom: [{ eq: "lorentz_transformation", assuming: "Simultaneous measurements in rest frame" }],
    },

    "relativistic_momentum": {
        status: "derived",
        conditions: "All velocities",
        derivesFrom: [{ eq: "lorentz_transformation", assuming: "momentum transforms as four-vector" }],
    },

    "relativistic_ke": {
        status: "derived",
        conditions: "All velocities",
        derivesFrom: [{ eq: "energy_momentum_relation", assuming: "T = E - mc²" }],
        forms: { integral: "K = \\int_0^v \\frac{mv}{(1 - v^2/c^2)^{3/2}} \\, dv" },
    },

    "relativistic_doppler": {
        status: "derived",
        conditions: "All velocities (includes transverse Doppler)",
        derivesFrom: [{ eq: "lorentz_transformation", assuming: "applied to light wave phase" }],
    },

    "relativistic_velocity_addition": {
        status: "derived",
        conditions: "All velocities",
        derivesFrom: [{ eq: "lorentz_transformation", assuming: "Composition of two boosts" }],
    },

    "lorentz_transformation": {
        status: "derived",
        conditions: "Inertial frames (flat spacetime)",
        statusNote: "Foundation of special relativity — replaces Galilean transformations",
        supersededBy: "einstein_field_equations",
        equivalentTo: ["spacetime_interval"],
    },

    "spacetime_interval": {
        status: "derived",
        conditions: "Flat spacetime",
        equivalentTo: ["lorentz_transformation"],
    },

    "four_momentum": {
        status: "derived",
        conditions: "All special relativity",
        derivesFrom: [{ eq: "energy_momentum_relation", assuming: "four-vector formulation" }],
    },


    // ─── General Relativity ─────────────────────────────────────

    "einstein_field_equations": {
        status: "postulate",
        conditions: "All gravitational phenomena",
        statusNote: "Einstein's 1915 theory; reduces to Newton in weak-field limit",
        incompatibleWith: ["newton_gravitation"],
    },

    "schwarzschild_radius": {
        status: "derived",
        conditions: "Non-rotating, uncharged black hole",
        derivesFrom: [{ eq: "schwarzschild_metric", assuming: "Event horizon condition g_tt = 0" }],
    },

    "gravitational_time_dilation": {
        status: "derived",
        conditions: "Schwarzschild spacetime",
        derivesFrom: [{ eq: "schwarzschild_metric", assuming: "Static observer" }],
    },

    "gravitational_redshift": {
        status: "derived",
        conditions: "Static gravitational field",
        derivesFrom: [{ eq: "schwarzschild_metric", assuming: "Photon escaping gravitational well" }],
    },

    "geodesic_equation": {
        status: "derived",
        conditions: "Free-falling particles in curved spacetime",
        derivesFrom: [{ eq: "einstein_field_equations", assuming: "Extremal proper time" }],
    },

    "perfect_fluid": {
        status: "definition",
        conditions: "Perfect fluid matter model",
        derivesFrom: [{ eq: "einstein_field_equations", assuming: "isotropic pressure, no viscosity" }],
    },

    "einstein_tensor_def": {
        status: "definition",
        derivesFrom: [{ eq: "einstein_field_equations", assuming: "contracted Riemann tensor" }],
    },

    "schwarzschild_metric": {
        status: "solution",
        conditions: "Vacuum, spherically symmetric, non-rotating mass",
        derivesFrom: [{ eq: "einstein_field_equations", assuming: "Spherical symmetry, vacuum T_μν = 0" }],
    },

    "gravitational_lensing": {
        status: "approximation",
        conditions: "Weak field, small deflection angles",
        derivesFrom: [{ eq: "schwarzschild_metric", assuming: "Weak field, light-like geodesic" }],
    },

    "stress_energy_conservation": {
        status: "derived",
        conditions: "All spacetimes in general relativity",
        derivesFrom: [{ eq: "einstein_field_equations", assuming: "Bianchi identity ∇_μ G^μν = 0" }],
    },


    // ─── Chemistry ──────────────────────────────────────────────

    "arrhenius": {
        status: "empirical",
        conditions: "Elementary reactions, single activation barrier",
        statusNote: "Semi-empirical — exponential form is derived but prefactor A is empirical",
        supersededBy: "eyring",
    },

    "nernst": {
        status: "derived",
        conditions: "Electrochemical cells, non-standard conditions",
        derivesFrom: [{ eq: "gibbs_free_energy", assuming: "Electrochemical equilibrium" }],
    },

    "gibbs_reaction": {
        status: "derived",
        derivesFrom: [{ eq: "gibbs_free_energy", assuming: "difference of products and reactants" }],
    },

    "equilibrium_gibbs": {
        status: "derived",
        derivesFrom: [{ eq: "gibbs_free_energy", assuming: "ΔG = 0 at equilibrium" }],
    },

    "beer_lambert": {
        status: "empirical",
        conditions: "Dilute solutions, monochromatic light",
        statusNote: "Fails at high concentrations, scattering media, fluorescent samples",
    },

    "ideal_gas_molecular": {
        status: "derived",
        equivalentTo: ["ideal_gas_law"],
        derivesFrom: [{ eq: "ideal_gas_law", assuming: "n = N/N_A, R = N_A k_B" }],
    },

    "faraday_electrolysis": {
        status: "empirical",
        statusNote: "Faraday's 1834 law; quantized charge transfer in electrochemistry",
        derivesFrom: [{ eq: "coulombs_law", assuming: "discrete charge carriers in electrolyte" }],
    },

    "raoults_law": {
        status: "empirical",
        conditions: "Ideal solutions (similar intermolecular forces)",
        statusNote: "Ideal solution limit; fails for non-ideal mixtures",
    },

    "eyring": {
        status: "derived",
        conditions: "Transition state theory",
        derivesFrom: [{ eq: "partition_function", assuming: "transition state theory" }],
    },

    "equilibrium_expression": {
        status: "definition",
        derivesFrom: [{ eq: "gibbs_free_energy", assuming: "at equilibrium ΔG = 0" }],
    },

    "absorbance_def": {
        status: "definition",
        derivesFrom: [{ eq: "beer_lambert", assuming: "A = -log₁₀(T)" }],
    },

    "beer_lambert_transmittance": {
        status: "derived",
        conditions: "Dilute solutions, monochromatic light",
        derivesFrom: [{ eq: "beer_lambert", assuming: "Exponential form of absorbance" }],
    },

    "henderson_hasselbalch": {
        status: "approximation",
        conditions: "Dilute buffer solutions, monoprotonic acid",
        derivesFrom: [{ eq: "equilibrium_expression", assuming: "weak acid buffer, dilute solution" }],
    },

    "hesss_law": {
        status: "derived",
        conditions: "All chemical reactions",
        statusNote: "Direct consequence of enthalpy being a state function",
        derivesFrom: [{ eq: "first_law_thermo", assuming: "enthalpy is a state function" }],
    },

    "grahams_law": {
        status: "derived",
        conditions: "Ideal gas effusion through small orifice",
        derivesFrom: [{ eq: "kinetic_theory_pressure", assuming: "effusion through small orifice" }],
    },

    "henry_law": {
        status: "empirical",
        conditions: "Dilute solutions, low partial pressure",
        statusNote: "Limiting case of Raoult's law for dilute solutes",
        supersededBy: "raoults_law",
    },

    "molar_mass_relation": {
        status: "definition",
        statusNote: "Bridges atomic mass unit and molar quantities via Avogadro's number",
        derivesFrom: [{ eq: "gas_constant_relation", assuming: "M = m × N_A" }],
    },


    // ─── Statistical Mechanics ──────────────────────────────────

    "partition_function": {
        status: "definition",
        conditions: "Canonical ensemble",
        statusNote: "Central object of equilibrium statistical mechanics",
        derivesFrom: [{ eq: "boltzmann_entropy", assuming: "summing over microstates" }],
    },

    "boltzmann_distribution": {
        status: "derived",
        conditions: "Thermal equilibrium",
        derivesFrom: [{ eq: "partition_function", assuming: "Maximum entropy at fixed energy" }],
    },

    "free_energy_partition": {
        status: "derived",
        conditions: "Canonical ensemble",
        derivesFrom: [{ eq: "partition_function", assuming: "Logarithmic bridge to thermodynamics" }],
    },

    "fermi_dirac": {
        status: "derived",
        conditions: "All fermions (half-integer spin)",
        incompatibleWith: ["bose_einstein"],
        derivesFrom: [{ eq: "partition_function", assuming: "indistinguishable fermions, Pauli exclusion" }],
    },

    "bose_einstein": {
        status: "derived",
        conditions: "All bosons (integer spin)",
        incompatibleWith: ["fermi_dirac"],
        derivesFrom: [{ eq: "partition_function", assuming: "indistinguishable bosons, no exclusion" }],
    },

    "stefan_boltzmann_derivation": {
        status: "derived",
        derivesFrom: [{ eq: "planck_law", assuming: "Integration over all frequencies and solid angles" }],
    },

    "sackur_tetrode": {
        status: "derived",
        conditions: "Ideal monatomic gas",
        derivesFrom: [{ eq: "partition_function", assuming: "ideal gas of indistinguishable particles" }],
    },

    "ideal_gas_microstates": {
        status: "derived",
        conditions: "Ideal gas, large N",
        derivesFrom: [{ eq: "partition_function", assuming: "classical ideal gas phase space" }],
    },

    "entropy_information_bridge": {
        status: "derived",
        conditions: "All probability distributions",
        statusNote: "Identical in form to Shannon entropy — revealing the deep link between physics and information",
        equivalentTo: ["boltzmann_entropy"],
        derivesFrom: [{ eq: "boltzmann_entropy", assuming: "generalized to probability distributions" }],
    },

    "grand_canonical": {
        status: "definition",
        conditions: "Open systems exchanging energy & particles",
        derivesFrom: [{ eq: "partition_function", assuming: "variable particle number, chemical potential" }],
    },

    "fluctuation_dissipation": {
        status: "derived",
        conditions: "Near-equilibrium systems",
        statusNote: "Connects thermal noise to dissipation; foundation of Nyquist theorem",
        derivesFrom: [{ eq: "boltzmann_distribution", assuming: "linear response near equilibrium" }],
    },

    "mean_free_path_def": {
        status: "derived",
        conditions: "Dilute gas, hard sphere model",
        derivesFrom: [{ eq: "kinetic_theory_pressure", assuming: "molecular collision geometry" }],
    },

    "nyquist_johnson": {
        status: "derived",
        conditions: "Thermal equilibrium, linear resistance",
        derivesFrom: [{ eq: "fluctuation_dissipation", assuming: "Linear resistance at thermal equilibrium" }],
    },

    "einstein_diffusion": {
        status: "derived",
        conditions: "Linear response, thermal equilibrium",
        derivesFrom: [{ eq: "fluctuation_dissipation", assuming: "Brownian particle in fluid" }],
    },


    // ─── Fluid Mechanics ────────────────────────────────────────

    "bernoulli": {
        status: "approximation",
        conditions: "Incompressible, inviscid, steady, irrotational",
        supersededBy: "navier_stokes",
        forms: { differential: "\\frac{\\partial \\vec{v}}{\\partial t} + (\\vec{v}\\!\\cdot\\!\\nabla)\\vec{v} = -\\frac{\\nabla P}{\\rho} + \\vec{g}" },
        formNotes: { differential: "Euler's equation — generalizes to unsteady inviscid flow" },
    },

    "continuity": {
        status: "approximation",
        conditions: "Incompressible flow (ρ = const)",
        derivesFrom: [{ eq: "navier_stokes", assuming: "mass conservation, incompressible flow" }],
        forms: { differential: "\\frac{\\partial \\rho}{\\partial t} + \\nabla \\cdot (\\rho \\vec{v}) = 0" },
        formNotes: { differential: "Full continuity equation for compressible flow" },
    },

    "reynolds_number": {
        status: "definition",
        derivesFrom: [{ eq: "navier_stokes", assuming: "ratio of inertial to viscous forces" }],
    },

    "stokes_law": {
        status: "approximation",
        conditions: "Re \ll 1 (creeping flow around spheres)",
        supersededBy: "navier_stokes",
    },

    "archimedes": {
        status: "derived",
        conditions: "Any fluid & immersed body",
        derivesFrom: [{ eq: "navier_stokes", assuming: "hydrostatic pressure gradient" }],
    },

    "poiseuille": {
        status: "derived",
        conditions: "Laminar, incompressible, Newtonian, fully developed pipe flow",
        derivesFrom: [{ eq: "navier_stokes", assuming: "Cylindrical symmetry, steady, fully developed" }],
    },

    "drag_force": {
        status: "empirical",
        conditions: "High Reynolds number",
        statusNote: "Phenomenological; drag coefficient C_d depends on shape and Reynolds number",
        derivesFrom: [{ eq: "navier_stokes", assuming: "dimensional analysis of turbulent drag" }],
    },

    "flow_rate_def": {
        status: "definition",
        derivesFrom: [{ eq: "continuity", assuming: "Q = Av from mass conservation" }],
    },

    "turbulent_transition": {
        status: "empirical",
        conditions: "Pipe flow",
        statusNote: "Re_crit ≈ 2300 for pipe flow; geometry-dependent",
        derivesFrom: [{ eq: "reynolds_number", assuming: "critical value for laminar-turbulent transition" }],
    },

    "navier_stokes": {
        status: "derived",
        conditions: "All viscous, Newtonian fluid flow",
        statusNote: "One of the Clay Millennium Prize Problems — existence and smoothness of solutions is unproven",
        derivesFrom: [{ eq: "newton_second_momentum", assuming: "Continuum hypothesis, Newtonian fluid" }],
    },

    "torricelli": {
        status: "derived",
        conditions: "Large reservoir, small orifice, no viscosity",
        supersededBy: "bernoulli",
        derivesFrom: [{ eq: "bernoulli", assuming: "Large reservoir, open orifice" }],
    },

    "young_laplace": {
        status: "derived",
        conditions: "Static curved interfaces",
        derivesFrom: [{ eq: "navier_stokes", assuming: "surface tension at curved interface" }],
    },

    "stokes_einstein": {
        status: "derived",
        conditions: "Spherical particles, low Re, dilute solution",
        derivesFrom: [
            { eq: "stokes_law", assuming: "combined with Einstein diffusion relation" },
            { eq: "einstein_diffusion", assuming: "Brownian sphere in viscous fluid" }
        ],
    },

    "fick_first_law": {
        status: "empirical",
        conditions: "Steady-state diffusion",
        statusNote: "Phenomenological diffusion law; analogous to Fourier's law for heat",
        derivesFrom: [{ eq: "boltzmann_distribution", assuming: "random walk of particles down concentration gradient" }],
        forms: { differential: "\\frac{\\partial c}{\\partial t} = D \\nabla^2 c" },
        formNotes: { differential: "Fick's Second Law — time-dependent diffusion" },
    },


    // ─── Optics & Waves ─────────────────────────────────────────

    "wave_equation": {
        status: "derived",
        conditions: "All linear wave phenomena",
        derivesFrom: [{ eq: "newton_second_law", assuming: "small displacement in elastic medium" }],
        forms: { differential: "\\frac{\\partial^2 u}{\\partial t^2} = v^2 \\nabla^2 u" },
        formNotes: { differential: "3D wave equation with Laplacian" },
    },

    "snells_law": {
        status: "derived",
        conditions: "Isotropic, homogeneous media",
        statusNote: "Originally empirical (1621); derivable from Maxwell's equations",
        derivesFrom: [{ eq: "em_wave_equation", assuming: "boundary conditions at interface" }],
    },

    "diffraction_grating": {
        status: "derived",
        conditions: "Multiple-slit interference, far field",
        derivesFrom: [{ eq: "wave_equation", assuming: "constructive interference from periodic slits" }],
    },

    "doppler_classical": {
        status: "approximation",
        conditions: "v \ll c (non-relativistic)",
        supersededBy: "relativistic_doppler",
    },

    "thin_lens": {
        status: "approximation",
        conditions: "Thin lens, paraxial rays",
        supersededBy: "lensmaker",
    },

    "malus_law": {
        status: "derived",
        conditions: "Polarized light through ideal polarizer",
        derivesFrom: [{ eq: "em_wave_equation", assuming: "linear polarization through analyzer" }],
    },

    "rayleigh_scattering": {
        status: "derived",
        conditions: "Particles much smaller than wavelength",
        statusNote: "Why the sky is blue",
        derivesFrom: [{ eq: "em_wave_equation", assuming: "scattering from particles ≪ wavelength" }],
    },

    "refractive_index_def": {
        status: "definition",
        derivesFrom: [{ eq: "em_wave_speed", assuming: "n = c/v in medium" }],
    },

    "rayleigh_jeans": {
        status: "approximation",
        conditions: "h\nu \ll k_BT (low frequency limit)",
        statusNote: "Its failure at high frequencies (ultraviolet catastrophe) launched quantum theory",
        supersededBy: "planck_law",
        incompatibleWith: ["planck_law"],
    },

    "brewster_angle": {
        status: "derived",
        conditions: "Plane interface, non-magnetic dielectrics",
        derivesFrom: [{ eq: "snells_law", assuming: "reflected + refracted rays perpendicular" }],
    },

    "rayleigh_criterion": {
        status: "derived",
        conditions: "Circular aperture, incoherent illumination",
        derivesFrom: [{ eq: "diffraction_grating", assuming: "circular aperture diffraction limit" }],
    },

    "double_slit": {
        status: "derived",
        conditions: "Coherent monochromatic source, far field",
        derivesFrom: [{ eq: "wave_equation", assuming: "two-slit interference pattern" }],
    },

    "lensmaker": {
        status: "derived",
        conditions: "Thin lens, paraxial rays",
        derivesFrom: [{ eq: "snells_law", assuming: "paraxial approximation, thin lens" }],
    },

    "total_internal_reflection": {
        status: "derived",
        conditions: "Interface from high-n to low-n medium",
        derivesFrom: [{ eq: "snells_law", assuming: "sin θ₂ = 1 (critical angle)" }],
    },

    "bragg_diffraction": {
        status: "derived",
        conditions: "X-rays on crystalline lattices",
        derivesFrom: [{ eq: "wave_equation", assuming: "constructive interference from crystal planes" }],
    },


    // ─── Nuclear & Particle Physics ─────────────────────────────

    "radioactive_decay": {
        status: "derived",
        conditions: "Constant decay probability per unit time",
        derivesFrom: [{ eq: "born_rule", assuming: "constant transition probability per unit time" }],
    },

    "half_life_relation": {
        status: "definition",
        derivesFrom: [{ eq: "radioactive_decay", assuming: "N(t₁/₂) = N₀/2" }],
    },

    "binding_energy": {
        status: "derived",
        conditions: "All nuclear binding",
        derivesFrom: [{ eq: "mass_energy", assuming: "Mass defect of nucleus" }],
    },

    "nuclear_radius": {
        status: "empirical",
        conditions: "All nuclei (approximate fit)",
        statusNote: "R = R₀ A^(1/3); R₀ ≈ 1.2-1.3 fm from electron scattering data",
        derivesFrom: [{ eq: "bethe_weizsacker", assuming: "nuclear density approximately constant" }],
    },

    "decay_activity": {
        status: "definition",
        derivesFrom: [{ eq: "radioactive_decay", assuming: "A = λN = -dN/dt" }],
    },

    "decay_by_half_life": {
        status: "derived",
        derivesFrom: [{ eq: "radioactive_decay", assuming: "Substituting t₁/₂ = ln2/λ" }],
    },

    "bethe_weizsacker": {
        status: "empirical",
        conditions: "Medium-to-heavy nuclei (A > 20)",
        statusNote: "Semi-empirical; liquid-drop model with quantum corrections",
        derivesFrom: [{ eq: "mass_energy", assuming: "nuclear binding modeled as liquid drop" }],
    },

    "q_value": {
        status: "derived",
        conditions: "All nuclear reactions",
        derivesFrom: [{ eq: "mass_energy", assuming: "Mass-energy conservation" }],
    },

    "geiger_nuttall": {
        status: "empirical",
        conditions: "Alpha decay only",
        statusNote: "Empirical fit; later explained by quantum tunneling through the Coulomb barrier",
        supersededBy: "tunnel_transmission",
    },

    "rutherford_scattering": {
        status: "derived",
        conditions: "Point-like projectile, pure Coulomb potential",
        derivesFrom: [{ eq: "coulombs_law", assuming: "Classical trajectory, point nucleus" }],
    },

    "cross_section_rate": {
        status: "derived",
        conditions: "All beam-target nuclear reactions",
        derivesFrom: [{ eq: "born_rule", assuming: "scattering probability from incident flux" }],
    },


    // ─── Astrophysics & Cosmology ───────────────────────────────

    "hubbles_law": {
        status: "empirical",
        conditions: "Nearby galaxies (z \ll 1)",
        statusNote: "Originally empirical (1929); now understood as the local linear limit of Friedmann expansion",
        supersededBy: "friedmann",
    },

    "stellar_luminosity": {
        status: "derived",
        conditions: "Blackbody star",
        derivesFrom: [{ eq: "stefan_boltzmann", assuming: "Spherical blackbody" }],
    },

    "friedmann": {
        status: "derived",
        conditions: "Homogeneous, isotropic universe (FLRW metric)",
        derivesFrom: [{ eq: "einstein_field_equations", assuming: "FLRW metric, homogeneity, isotropy" }],
    },

    "hawking_temperature": {
        status: "derived",
        conditions: "Schwarzschild black hole",
        statusNote: "Hawking 1974; not yet experimentally confirmed",
        derivesFrom: [{ eq: "einstein_field_equations", assuming: "quantum fields in curved spacetime" }],
    },

    "gravitational_wave_power": {
        status: "derived",
        conditions: "Weak-field, slow-motion binary",
        derivesFrom: [{ eq: "einstein_field_equations", assuming: "Linearized GR, quadrupole radiation" }],
    },

    "chandrasekhar_mass": {
        status: "derived",
        conditions: "Fully degenerate, non-rotating white dwarf",
        statusNote: "Chandrasekhar 1930; maximum mass for white dwarf ≈ 1.4 M☉",
        derivesFrom: [{ eq: "fermi_dirac", assuming: "electron degeneracy pressure vs gravity" }],
    },

    "inverse_square_intensity": {
        status: "derived",
        conditions: "Isotropic point source",
        derivesFrom: [{ eq: "sphere_surface", assuming: "energy conservation over expanding sphere" }],
    },

    "hubble_scale": {
        status: "definition",
        derivesFrom: [{ eq: "friedmann", assuming: "defined as ȧ/a" }],
    },

    "dark_energy_density": {
        status: "derived",
        conditions: "ΛCDM cosmological constant model",
        derivesFrom: [{ eq: "friedmann", assuming: "cosmological constant Λ" }],
    },

    "cosmological_redshift": {
        status: "derived",
        conditions: "FLRW expanding universe",
        derivesFrom: [{ eq: "friedmann", assuming: "Expanding FLRW metric" }],
    },

    "jeans_mass": {
        status: "derived",
        conditions: "Uniform density, isothermal gas cloud",
        derivesFrom: [
            { eq: "gravitational_pe_general", assuming: "gravitational collapse vs thermal pressure" },
            { eq: "ideal_gas_law", assuming: "thermal support in gas cloud" }
        ],
    },

    "tov_equation": {
        status: "derived",
        conditions: "Spherically symmetric, static star in GR",
        derivesFrom: [{ eq: "einstein_field_equations", assuming: "Static, spherically symmetric perfect fluid" }],
    },

    "eddington_luminosity": {
        status: "derived",
        conditions: "Spherically symmetric, electron scattering opacity",
        derivesFrom: [{ eq: "newton_second_law", assuming: "radiation pressure balanced by gravity" }],
    },

    "saha_equation": {
        status: "derived",
        conditions: "Thermal equilibrium ionization",
        derivesFrom: [{ eq: "boltzmann_distribution", assuming: "ionization equilibrium in plasma" }],
    },


    // ─── Information Theory ─────────────────────────────────────

    "shannon_entropy": {
        status: "definition",
        statusNote: "Shannon 1948; foundational measure of information content",
    },

    "landauer_principle": {
        status: "derived",
        conditions: "Reversible computation at thermal equilibrium",
        statusNote: "Information is physical — erasing a bit dissipates energy",
        derivesFrom: [
            { eq: "shannon_entropy", assuming: "erasure of 1 bit requires k_B T ln 2 energy" },
            { eq: "boltzmann_entropy", assuming: "thermodynamic cost of information erasure" }
        ],
    },

    "bekenstein_hawking": {
        status: "derived",
        conditions: "Black hole event horizons",
        statusNote: "Unites gravity, quantum mechanics, thermodynamics, and information",
        derivesFrom: [
            { eq: "einstein_field_equations", assuming: "black hole thermodynamics" },
            { eq: "shannon_entropy", assuming: "entropy proportional to horizon area" }
        ],
    },

    "shannon_hartley": {
        status: "derived",
        conditions: "Additive white Gaussian noise channel",
        derivesFrom: [{ eq: "shannon_entropy", assuming: "Gaussian noise channel" }],
    },

    "kl_divergence": {
        status: "definition",
        derivesFrom: [{ eq: "shannon_entropy", assuming: "relative entropy between distributions" }],
    },

    "mutual_information": {
        status: "derived",
        derivesFrom: [{ eq: "shannon_entropy", assuming: "I(X;Y) = H(X) + H(Y) - H(X,Y)" }],
    },

    "data_processing_inequality": {
        status: "derived",
        conditions: "Markov chains X → Y → Z",
        derivesFrom: [{ eq: "mutual_information", assuming: "Markov chain X → Y → Z" }],
    },


    // ─── Derived Constants & Relations ──────────────────────────

    "gas_constant_relation": {
        status: "definition",
        statusNote: "Bridges macroscopic R and microscopic k_B via Avogadro's number",
        derivesFrom: [{ eq: "ideal_gas_law", assuming: "R = N_A k_B from PV = Nk_BT" }],
    },

    "faraday_constant_relation": {
        status: "definition",
        statusNote: "Connects Avogadro's number to elementary charge; key to electrochemistry",
        derivesFrom: [{ eq: "faraday_electrolysis", assuming: "charge per mole of electrons" }],
    },

    "coulomb_constant_relation": {
        status: "definition",
        derivesFrom: [{ eq: "coulombs_law", assuming: "k_e = 1/(4πε₀)" }],
    },

    "reduced_planck_relation": {
        status: "definition",
        statusNote: "ℏ = h/(2π); natural unit for angular momentum in quantum mechanics",
        derivesFrom: [{ eq: "planck_einstein", assuming: "ℏ = h/(2π) for angular frequency" }],
    },

    "rydberg_from_constants": {
        status: "derived",
        derivesFrom: [{ eq: "hydrogen_energy", assuming: "expressed in fundamental constants" }],
    },

    "wien_from_constants": {
        status: "derived",
        derivesFrom: [{ eq: "planck_law", assuming: "derivative set to zero" }],
    },

    "bohr_magneton": {
        status: "derived",
        derivesFrom: [{ eq: "magnetic_moment_orbital", assuming: "electron with orbital angular momentum ℏ/2" }],
    },

    "planck_units": {
        status: "definition",
        statusNote: "The smallest meaningful length scale — where quantum mechanics and gravity meet",
        componentOf: "planck_natural_units",
    },

    "planck_temperature": {
        status: "definition",
        componentOf: "planck_natural_units",
    },

    "planck_mass": {
        status: "definition",
        componentOf: "planck_natural_units",
    },

    "rydberg_fine_structure": {
        status: "derived",
        derivesFrom: [{ eq: "rydberg_from_constants", assuming: "rewritten using α" }],
    },

    "bohr_radius_fine_structure": {
        status: "derived",
        derivesFrom: [{ eq: "bohr_radius", assuming: "rewritten using α" }],
    },

    // ─── NEW EQUATIONS — Batch 2 Meta ───────────────────────────────────────

    // Classical Mechanics
    "hookes_law_stress_strain": {
        status: "empirical",
        conditions: "Elastic limit not exceeded, isotropic material",
        derivesFrom: [{ eq: "hookes_law", assuming: "continuum limit of spring law" }],
    },
    "bulk_modulus_def": {
        status: "definition",
    },
    "poisson_ratio_def": {
        status: "definition",
        statusNote: "Typically 0.2–0.5 for most materials; 0.5 for incompressible",
    },
    "parallel_axis_theorem": {
        status: "derived",
        derivesFrom: [{ eq: "rotational_ke", assuming: "shifted rotation axis" }],
    },
    "damped_oscillator": {
        status: "derived",
        conditions: "Linear damping, linear restoring force",
        derivesFrom: [{ eq: "newton_second_law", assuming: "Hooke's law + linear drag" }],
    },
    "reduced_mass": {
        status: "derived",
        derivesFrom: [{ eq: "newton_second_law", assuming: "two-body → one-body transformation" }],
    },

    // Electromagnetism
    "maxwell_displacement_current": {
        status: "postulate",
        statusNote: "Maxwell's key theoretical prediction — completing the Ampère–Maxwell equation",
        equivalentTo: ["ampere_maxwell"],
    },
    "coulombs_law_vector": {
        status: "approximation",
        conditions: "Point charges, electrostatics",
        supersededBy: "gauss_law",
        equivalentTo: ["coulombs_law"],
    },
    "ohms_law_field": {
        status: "empirical",
        conditions: "Ohmic materials, linear response",
        equivalentTo: ["ohms_law"],
    },
    "polarization_field": {
        status: "derived",
        conditions: "Linear, isotropic dielectric",
        derivesFrom: [{ eq: "gauss_law", assuming: "linear dielectric medium" }],
    },
    "energy_density_em": {
        status: "derived",
        derivesFrom: [{ eq: "poynting_vector", assuming: "energy conservation in EM fields" }],
    },
    "rc_time_constant": {
        status: "derived",
        derivesFrom: [{ eq: "ohms_law", assuming: "capacitor charging through resistor" }],
    },

    // Quantum Mechanics
    "pauli_exclusion": {
        status: "postulate",
        statusNote: "Fundamental postulate of quantum mechanics — explains atomic structure and the stability of matter",
        incompatibleWith: ["bose_einstein"],
    },
    "angular_momentum_quantization": {
        status: "derived",
        derivesFrom: [{ eq: "schrodinger_ti", assuming: "central potential, spherical harmonics" }],
    },
    "spin_z_quantization": {
        status: "postulate",
        statusNote: "Intrinsic quantum property with no classical analog",
        derivesFrom: [{ eq: "dirac_equation", assuming: "non-relativistic limit" }],
    },
    "wkb_approximation": {
        status: "approximation",
        conditions: "Slowly varying potential, λ ≪ scale of V(x)",
        derivesFrom: [{ eq: "schrodinger_ti", assuming: "semiclassical limit ℏ → 0" }],
    },
    "variational_principle": {
        status: "derived",
        statusNote: "The most widely used approximation method in quantum chemistry",
        derivesFrom: [{ eq: "schrodinger_ti", assuming: "completeness of Hilbert space" }],
    },
    "density_of_states_3d": {
        status: "derived",
        derivesFrom: [{ eq: "particle_in_box", assuming: "large volume limit, 3D" }],
    },

    // Special & General Relativity
    "proper_time_def": {
        status: "definition",
        derivesFrom: [{ eq: "time_dilation", assuming: "infinitesimal time interval" }],
    },
    "relativistic_energy_full": {
        status: "derived",
        derivesFrom: [{ eq: "energy_momentum_relation", assuming: "expanding with rest mass" }],
        equivalentTo: ["mass_energy"],
    },
    "kerr_metric": {
        status: "solution",
        statusNote: "Exact rotating black hole solution — the most astrophysically relevant metric",
        derivesFrom: [{ eq: "einstein_field_equations", assuming: "axially symmetric vacuum" }],
    },
    "grav_wave_strain": {
        status: "derived",
        conditions: "Weak-field, slow-motion (quadrupole approximation)",
        derivesFrom: [{ eq: "einstein_field_equations", assuming: "linearized gravity" }],
    },
    "frame_dragging": {
        status: "derived",
        derivesFrom: [{ eq: "kerr_metric", assuming: "weak-field limit, slow rotation" }],
    },

    // Thermodynamics
    "joule_thomson": {
        status: "definition",
        statusNote: "Measured experimentally; sign determines cooling vs heating in throttling",
    },
    "maxwell_relation_sp": {
        status: "derived",
        derivesFrom: [{ eq: "gibbs_free_energy", assuming: "exactness of dG" }],
        equivalentTo: ["maxwell_relation_tv"],
    },
    "gibbs_duhem": {
        status: "derived",
        statusNote: "Constrains intensive variables — can't vary T, P, and all μᵢ independently",
        derivesFrom: [{ eq: "gibbs_free_energy", assuming: "Euler's theorem for homogeneous functions" }],
    },
    "clapeyron_equation": {
        status: "derived",
        statusNote: "Exact — no approximations. Clausius–Clapeyron approximates this.",
        derivesFrom: [{ eq: "gibbs_free_energy", assuming: "two-phase equilibrium dG₁ = dG₂" }],
    },
    "ideal_gas_adiabatic": {
        status: "derived",
        conditions: "Ideal gas, reversible adiabatic process",
        derivesFrom: [{ eq: "first_law_thermo", assuming: "Q=0, ideal gas" }],
    },

    // Statistical Mechanics
    "detailed_balance": {
        status: "postulate",
        statusNote: "Microscopic reversibility at equilibrium",
        derivesFrom: [{ eq: "boltzmann_distribution", assuming: "transition rates between microstates" }],
    },
    "ising_partition": {
        status: "solution",
        conditions: "1D nearest-neighbor interactions, zero external field",
        derivesFrom: [{ eq: "partition_function", assuming: "1D Ising Hamiltonian" }],
    },
    "equipartition_general": {
        status: "derived",
        conditions: "Classical regime, quadratic Hamiltonian",
        derivesFrom: [{ eq: "boltzmann_distribution", assuming: "Gaussian integral over quadratic terms" }],
        equivalentTo: ["equipartition"],
    },
    "debye_heat_capacity": {
        status: "derived",
        conditions: "Solid at low to moderate temperatures",
        supersededBy: null,
        derivesFrom: [{ eq: "bose_einstein", assuming: "phonon spectrum with Debye cutoff" }],
    },
    "virial_expansion": {
        status: "derived",
        derivesFrom: [{ eq: "grand_canonical", assuming: "cluster expansion of interactions" }],
    },

    // Fluid Mechanics
    "euler_fluid": {
        status: "derived",
        conditions: "Inviscid flow, no body forces other than gravity",
        supersededBy: "navier_stokes",
        derivesFrom: [{ eq: "newton_second_law", assuming: "continuum fluid, no viscosity" }],
    },
    "kelvin_circulation": {
        status: "derived",
        conditions: "Inviscid, barotropic fluid, conservative body forces",
        derivesFrom: [{ eq: "euler_fluid", assuming: "material derivative of circulation integral" }],
    },
    "mach_number": {
        status: "definition",
        statusNote: "Ma < 1 subsonic, Ma > 1 supersonic, Ma > 5 hypersonic",
    },
    "vorticity_equation": {
        status: "derived",
        derivesFrom: [{ eq: "navier_stokes", assuming: "curl of momentum equation" }],
    },
    "capillary_rise": {
        status: "derived",
        conditions: "Narrow tube, contact angle known",
        derivesFrom: [{ eq: "young_laplace", assuming: "balance with hydrostatic pressure" }],
    },

    // Optics & Waves
    "huygens_fresnel": {
        status: "postulate",
        statusNote: "Later justified rigorously by the Kirchhoff diffraction integral",
    },
    "abbe_diffraction_limit": {
        status: "derived",
        derivesFrom: [{ eq: "double_slit", assuming: "circular aperture, first minimum" }],
    },
    "fresnel_reflection": {
        status: "derived",
        conditions: "Normal incidence, non-absorbing media",
        derivesFrom: [{ eq: "em_wave_equation", assuming: "boundary conditions at interface" }],
    },
    "standing_wave": {
        status: "derived",
        derivesFrom: [{ eq: "wave_equation", assuming: "fixed endpoints, superposition" }],
    },
    "group_velocity": {
        status: "definition",
        statusNote: "Equals phase velocity only in non-dispersive media",
    },

    // Nuclear Physics
    "fermi_four_point": {
        status: "approximation",
        conditions: "Low energy β-decay, four-fermion contact interaction",
        statusNote: "Superseded by electroweak theory (W boson exchange) at high energies",
        derivesFrom: [{ eq: "fermis_golden_rule", assuming: "four-fermion contact Hamiltonian" }],
    },
    "nuclear_shell_model": {
        status: "empirical",
        statusNote: "Explains magic numbers but requires spin-orbit coupling not predicted from first principles",
    },
    "liquid_drop_energy": {
        status: "empirical",
        equivalentTo: ["bethe_weizsacker"],
        statusNote: "Five terms with empirically fitted coefficients — accurate to ~1% for most nuclei",
    },
    "nuclear_fusion_rate": {
        status: "derived",
        derivesFrom: [{ eq: "cross_section_rate", assuming: "thermal velocity distribution of plasma ions" }],
    },

    // Astrophysics
    "virial_theorem_astro": {
        status: "derived",
        conditions: "Gravitationally bound system in quasi-static equilibrium",
        derivesFrom: [{ eq: "virial_theorem", assuming: "gravitational potential, time-averaged" }],
    },
    "mass_luminosity": {
        status: "empirical",
        conditions: "Main-sequence stars only",
        statusNote: "Exponent varies from ~3 for massive stars to ~4 for solar-type",
    },
    "schwarzschild_isco": {
        status: "derived",
        derivesFrom: [{ eq: "schwarzschild_metric", assuming: "circular orbit stability analysis" }],
    },
    "bondi_accretion": {
        status: "derived",
        conditions: "Spherically symmetric, steady-state, adiabatic",
        derivesFrom: [{ eq: "euler_fluid", assuming: "spherical symmetry + gravitational potential" }],
    },
    "tully_fisher": {
        status: "empirical",
        statusNote: "Key distance indicator; slope consistent with dark matter halos",
    },

    // Chemistry
    "rate_law_general": {
        status: "empirical",
        statusNote: "Orders m, n determined experimentally — not from balanced equation",
    },
    "integrated_first_order": {
        status: "derived",
        derivesFrom: [{ eq: "rate_law_general", assuming: "first order (m=1, single reactant)" }],
    },
    "vant_hoff": {
        status: "derived",
        derivesFrom: [{ eq: "equilibrium_gibbs", assuming: "temperature dependence of ΔG°" }],
    },
    "debye_huckel": {
        status: "derived",
        conditions: "Dilute electrolyte solutions (ionic strength < 0.01 M)",
        derivesFrom: [{ eq: "coulombs_law", assuming: "Poisson–Boltzmann equation, linearized" }],
    },
    "osmotic_pressure": {
        status: "derived",
        conditions: "Dilute, ideal solution",
        derivesFrom: [{ eq: "chemical_potential_def", assuming: "semipermeable membrane equilibrium" }],
    },

    // Information Theory
    "channel_capacity_binary": {
        status: "derived",
        derivesFrom: [{ eq: "shannon_hartley", assuming: "binary symmetric channel model" }],
    },
    "source_coding_theorem": {
        status: "derived",
        statusNote: "The converse of the noiseless coding theorem — compression cannot beat entropy",
        derivesFrom: [{ eq: "shannon_entropy", assuming: "optimal prefix-free codes" }],
    },
    "fisher_information": {
        status: "definition",
        statusNote: "Foundation of the Cramér–Rao bound — the precision limit of all estimators",
    },
    "holographic_entropy": {
        status: "derived",
        statusNote: "Suggests spacetime and gravity emerge from information-theoretic principles",
        derivesFrom: [{ eq: "bekenstein_hawking", assuming: "generalized to arbitrary regions" }],
    },

    // Meta / Cross-Field
    "dimensional_analysis": {
        status: "derived",
        statusNote: "Purely mathematical result from the structure of physical dimensions",
    },
    "noether_theorem": {
        status: "derived",
        statusNote: "The deepest structural theorem in theoretical physics — symmetry ↔ conservation",
        derivesFrom: [{ eq: "euler_lagrange", assuming: "continuous symmetry of Lagrangian" }],
    },
    "liouville_theorem": {
        status: "derived",
        derivesFrom: [{ eq: "hamiltons_equations", assuming: "continuity equation in phase space" }],
    },
    "action_principle": {
        status: "postulate",
        statusNote: "The most fundamental formulation of physics — all classical and quantum theories derive from an action",
        equivalentTo: ["euler_lagrange", "hamiltons_equations"],
    },

    // ─── Orphan-Fix Equations META ──────────────────────────────────────────

    "angular_momentum_z": {
        status: "derived",
        derivesFrom: [{ eq: "schrodinger_ti", assuming: "spherical harmonics as eigenfunctions of L_z" }],
    },
    "zeeman_energy": {
        status: "derived",
        derivesFrom: [{ eq: "angular_momentum_z", assuming: "perturbation by uniform magnetic field" }],
    },
    "selection_rules_dipole": {
        status: "derived",
        statusNote: "Arise from symmetry properties of the electric dipole matrix element",
        derivesFrom: [{ eq: "angular_momentum_z", assuming: "parity and angular momentum algebra" }],
    },
    "impact_parameter_rutherford": {
        status: "derived",
        derivesFrom: [{ eq: "rutherford_scattering", assuming: "hyperbolic Coulomb trajectory" }],
    },
    "cross_section_impact_param": {
        status: "derived",
        statusNote: "General geometric relation valid for any central-force scattering",
        derivesFrom: [{ eq: "newton_second_law", assuming: "central-force scattering geometry" }],
    },
    "keplerian_orbit": {
        status: "derived",
        derivesFrom: [{ eq: "newton_gravitation", assuming: "two-body problem in center-of-mass frame" }],
    },
    "orbital_eccentricity_energy": {
        status: "derived",
        derivesFrom: [{ eq: "keplerian_orbit", assuming: "conservation of energy and angular momentum" }],
    },
    "vis_viva": {
        status: "derived",
        statusNote: "Energy conservation for Keplerian orbits — the most useful equation in orbital mechanics",
        derivesFrom: [{ eq: "newton_gravitation", assuming: "conservation of total mechanical energy" }],
    },
    "hubble_redshift": {
        status: "empirical",
        statusNote: "Observational law that launched modern cosmology",
    },
    "radiative_transfer": {
        status: "derived",
        derivesFrom: [{ eq: "optical_depth_definition", assuming: "pure absorption, no emission" }],
    },
    "optical_depth_definition": {
        status: "definition",
        statusNote: "Defines the dimensionless measure of opacity along a line of sight",
    },
    "elastic_moduli_relation": {
        status: "derived",
        statusNote: "Consequence of isotropy — reduces 21 elastic constants to just 2",
        derivesFrom: [{ eq: "hookes_law_stress_strain", assuming: "isotropic linear-elastic material" }],
    },
    "shear_stress_strain": {
        status: "derived",
        derivesFrom: [{ eq: "hookes_law_stress_strain", assuming: "pure shear deformation" }],
    },
    "bulk_youngs_poisson": {
        status: "derived",
        derivesFrom: [{ eq: "elastic_moduli_relation", assuming: "isotropic material under hydrostatic stress" }],
    },
    "curie_law": {
        status: "empirical",
        statusNote: "First quantitative law of magnetism — later derived from statistical mechanics of magnetic dipoles",
    },
    "magnetization_moment": {
        status: "definition",
        statusNote: "Defines the total magnetic moment of a uniformly magnetized body",
    },
    "drude_resistivity": {
        status: "derived",
        statusNote: "Classical model — surprisingly good for metals despite ignoring quantum effects",
        derivesFrom: [{ eq: "newton_second_law", assuming: "free-electron gas with random collisions" }],
    },
    "electric_dipole_radiation": {
        status: "derived",
        derivesFrom: [{ eq: "maxwell_displacement_current", assuming: "oscillating point dipole, far-field limit" }],
    },
    "spacetime_interval_proper": {
        status: "derived",
        statusNote: "The invariant interval is the geometric foundation of special relativity",
        derivesFrom: [{ eq: "lorentz_transformation", assuming: "Minkowski metric signature (−,+,+,+)" }],
    },
    "landau_free_energy": {
        status: "postulate",
        statusNote: "Phenomenological — the coefficients encode microscopic physics without requiring a specific model",
    },
    "order_parameter_scaling": {
        status: "derived",
        derivesFrom: [{ eq: "landau_free_energy", assuming: "minimization of F near T_c" }],
    },
    "correlation_length_divergence": {
        status: "derived",
        statusNote: "Universality: the exponent ν depends only on dimensionality and symmetry class",
        derivesFrom: [{ eq: "landau_free_energy", assuming: "Gaussian fluctuation expansion" }],
    },
    "ornstein_zernike_correlation": {
        status: "derived",
        derivesFrom: [{ eq: "correlation_length_divergence", assuming: "Fourier transform of susceptibility" }],
    },
    "mean_free_path_kinetic": {
        status: "derived",
        derivesFrom: [{ eq: "ideal_gas_molecular", assuming: "hard-sphere collision model" }],
    },
    "displacement_field": {
        status: "definition",
        statusNote: "Separates free from bound charge contributions in macroscopic electrostatics",
    },
    "compressibility_definition": {
        status: "definition",
        statusNote: "Fundamental thermodynamic response function — diverges at critical points",
    },
    "friis_formula": {
        status: "derived",
        statusNote: "Foundation of link-budget analysis in telecommunications",
        derivesFrom: [{ eq: "coulombs_law", assuming: "free-space propagation, matched antennas" }],
    },

};
