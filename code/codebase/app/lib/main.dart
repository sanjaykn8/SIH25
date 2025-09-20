import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'enhanced_login_screen.dart';
import 'firebase_options.dart';
import 'dart:math';
import 'dart:async';
import 'dart:ui';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);

  // Set system UI overlay style
  SystemChrome.setSystemUIOverlayStyle(
    SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
      systemNavigationBarColor: Colors.transparent,
    ),
  );

  runApp(MicroplasticDetectorApp());
}

class MicroplasticDetectorApp extends StatelessWidget {
  const MicroplasticDetectorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Microplastic Detector',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
        appBarTheme: AppBarTheme(
          backgroundColor: Colors.transparent,
          elevation: 0,
          systemOverlayStyle: SystemUiOverlayStyle.light,
        ),
        textTheme: TextTheme(
          headlineLarge: TextStyle(
            fontWeight: FontWeight.bold,
            letterSpacing: 1.2,
          ),
        ),
      ),
      home: AuthenticationWrapper(),
      routes: {
        '/login': (context) => EnhancedLoginScreen(), //change
        '/dashboard': (context) => DashboardScreen(),
      },
    );
  }
}

class AuthenticationWrapper extends StatelessWidget {
  const AuthenticationWrapper({super.key});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<User?>(
      stream: FirebaseAuth.instance.authStateChanges(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return UltraCreativeSplashScreen();
        }

        if (snapshot.hasData && snapshot.data != null) {
          return DashboardScreen();
        }

        return UltraCreativeSplashScreen();
      },
    );
  }
}

class UltraCreativeSplashScreen extends StatefulWidget {
  const UltraCreativeSplashScreen({super.key});

  @override
  _UltraCreativeSplashScreenState createState() =>
      _UltraCreativeSplashScreenState();
}

class _UltraCreativeSplashScreenState extends State<UltraCreativeSplashScreen>
    with TickerProviderStateMixin {
  // Animation Controllers
  late AnimationController _mainController;
  late AnimationController _particleController;
  late AnimationController _waveController;
  late AnimationController _logoController;
  late AnimationController _textController;
  late AnimationController _loadingController;
  late AnimationController _liquidController;
  late AnimationController _morphController;
  late AnimationController _glowController;
  late AnimationController _orbitController;

  // Animations
  late Animation<double> _fadeAnimation;
  late Animation<double> _scaleAnimation;
  late Animation<double> _slideAnimation;
  late Animation<double> _rotationAnimation;
  late Animation<double> _logoBreathAnimation;
  late Animation<double> _textRevealAnimation;
  late Animation<double> _loadingAnimation;
  late Animation<double> _liquidWaveAnimation;
  late Animation<double> _morphAnimation;
  late Animation<double> _glowAnimation;
  late Animation<double> _orbitAnimation;
  late Animation<Color?> _colorTransitionAnimation;

  // Visual Elements
  List<EnhancedParticle> particles = [];
  List<LiquidBubble> bubbles = [];
  List<OrbitingElement> orbitingElements = [];
  List<MorphingShape> morphingShapes = [];

  // Loading States
  List<String> loadingTexts = [
    'Initializing Quantum Sensors...',
    'Connecting Neural Networks...',
    'Calibrating Nano-Detection...',
    'Loading AI Algorithms...',
    'Synchronizing with Cloud...',
    'Activating Bio-Scanner...',
    'Preparing Analysis Engine...',
    'Ready for Detection!',
  ];
  int currentLoadingIndex = 0;
  Timer? _loadingTextTimer;

  @override
  void initState() {
    super.initState();
    _initializeAnimations();
    _generateVisualElements();
    _startAnimationSequence();
    _startLoadingTextCycle();
    _navigateAfterDelay();
  }

  void _initializeAnimations() {
    // Primary Controllers
    _mainController = AnimationController(
      duration: Duration(milliseconds: 4500),
      vsync: this,
    );

    _particleController = AnimationController(
      duration: Duration(seconds: 30),
      vsync: this,
    )..repeat();

    _waveController = AnimationController(
      duration: Duration(seconds: 5),
      vsync: this,
    )..repeat();

    _logoController = AnimationController(
      duration: Duration(milliseconds: 3000),
      vsync: this,
    );

    _textController = AnimationController(
      duration: Duration(milliseconds: 2500),
      vsync: this,
    );

    _loadingController = AnimationController(
      duration: Duration(milliseconds: 1500),
      vsync: this,
    )..repeat();

    _liquidController = AnimationController(
      duration: Duration(seconds: 10),
      vsync: this,
    )..repeat();

    _morphController = AnimationController(
      duration: Duration(seconds: 15),
      vsync: this,
    )..repeat();

    _glowController = AnimationController(
      duration: Duration(seconds: 4),
      vsync: this,
    )..repeat(reverse: true);

    _orbitController = AnimationController(
      duration: Duration(seconds: 20),
      vsync: this,
    )..repeat();

    // Advanced Animations
    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _mainController,
        curve: Interval(0.1, 0.6, curve: Curves.easeOutQuart),
      ),
    );

    _scaleAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _logoController, curve: Curves.elasticOut),
    );

    _slideAnimation = Tween<double>(begin: 120.0, end: 0.0).animate(
      CurvedAnimation(parent: _textController, curve: Curves.easeOutExpo),
    );

    _rotationAnimation = Tween<double>(begin: 0.0, end: 2 * pi).animate(
      CurvedAnimation(parent: _mainController, curve: Curves.easeInOutCubic),
    );

    _logoBreathAnimation = Tween<double>(begin: 0.92, end: 1.12).animate(
      CurvedAnimation(parent: _waveController, curve: Curves.easeInOutSine),
    );

    _textRevealAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _textController, curve: Curves.easeOutExpo),
    );

    _loadingAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _loadingController, curve: Curves.easeInOutCubic),
    );

    _liquidWaveAnimation = Tween<double>(
      begin: 0.0,
      end: 2 * pi,
    ).animate(_liquidController);

    _morphAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _morphController, curve: Curves.easeInOutSine),
    );

    _glowAnimation = Tween<double>(begin: 0.3, end: 1.2).animate(
      CurvedAnimation(parent: _glowController, curve: Curves.easeInOutSine),
    );

    _orbitAnimation = Tween<double>(
      begin: 0.0,
      end: 2 * pi,
    ).animate(_orbitController);

    _colorTransitionAnimation =
        ColorTween(begin: Color(0xFF0D47A1), end: Color(0xFF00E676)).animate(
      CurvedAnimation(
        parent: _morphController,
        curve: Curves.easeInOutSine,
      ),
    );
  }

  void _generateVisualElements() {
    final random = Random();

    // Generate enhanced particles with complex behaviors
    for (int i = 0; i < 60; i++) {
      particles.add(
        EnhancedParticle(
          x: random.nextDouble(),
          y: random.nextDouble(),
          size: random.nextDouble() * 8 + 2,
          speed: random.nextDouble() * 1.2 + 0.3,
          color: Color.lerp(
            Colors.blue.withOpacity(0.4),
            Colors.cyan.withOpacity(0.9),
            random.nextDouble(),
          )!,
          oscillation: random.nextDouble() * 3 + 1,
          phase: random.nextDouble() * 2 * pi,
          trailLength: random.nextInt(5) + 3,
        ),
      );
    }

    // Generate liquid bubbles with organic movement
    for (int i = 0; i < 20; i++) {
      bubbles.add(
        LiquidBubble(
          x: random.nextDouble(),
          y: random.nextDouble(),
          size: random.nextDouble() * 60 + 30,
          speed: random.nextDouble() * 0.4 + 0.1,
          color: Color.lerp(
            Colors.blue.withOpacity(0.05),
            Colors.cyan.withOpacity(0.2),
            random.nextDouble(),
          )!,
          pulsePhase: random.nextDouble() * 2 * pi,
        ),
      );
    }

    // Generate orbiting elements
    for (int i = 0; i < 6; i++) {
      orbitingElements.add(
        OrbitingElement(
          centerX: 0.5,
          centerY: 0.4,
          radius: (i + 1) * 40.0 + 80,
          speed: 0.3 + i * 0.1,
          size: 12 - i * 1.5,
          phase: i * pi / 3,
          color: Color.lerp(
            Colors.blue.withOpacity(0.6),
            Colors.cyan.withOpacity(0.8),
            i / 6.0,
          )!,
        ),
      );
    }

    // Generate morphing shapes
    for (int i = 0; i < 4; i++) {
      morphingShapes.add(
        MorphingShape(
          x: random.nextDouble(),
          y: random.nextDouble(),
          size: random.nextDouble() * 120 + 60,
          speed: random.nextDouble() * 0.1 + 0.05,
          morphPhase: random.nextDouble() * 2 * pi,
          shapeType: i % 3,
        ),
      );
    }
  }

  void _startAnimationSequence() {
    // Sophisticated staggered animation sequence
    Future.delayed(Duration(milliseconds: 300), () {
      _logoController.forward();
    });

    Future.delayed(Duration(milliseconds: 1000), () {
      _mainController.forward();
    });

    Future.delayed(Duration(milliseconds: 1800), () {
      _textController.forward();
    });
  }

  void _startLoadingTextCycle() {
    _loadingTextTimer = Timer.periodic(Duration(milliseconds: 700), (timer) {
      if (mounted) {
        setState(() {
          currentLoadingIndex = (currentLoadingIndex + 1) % loadingTexts.length;
        });
      }
    });
  }

  void _navigateAfterDelay() {
    Future.delayed(Duration(milliseconds: 6000), () {
      if (mounted) {
        final user = FirebaseAuth.instance.currentUser;
        if (user != null) {
          Navigator.pushReplacementNamed(context, '/dashboard');
        } else {
          Navigator.pushReplacementNamed(context, '/login');
        }
      }
    });
  }

  @override
  void dispose() {
    _mainController.dispose();
    _particleController.dispose();
    _waveController.dispose();
    _logoController.dispose();
    _textController.dispose();
    _loadingController.dispose();
    _liquidController.dispose();
    _morphController.dispose();
    _glowController.dispose();
    _orbitController.dispose();
    _loadingTextTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: AnimatedBuilder(
        animation: Listenable.merge([
          _colorTransitionAnimation,
          _morphAnimation,
        ]),
        builder: (context, child) {
          return Container(
            width: double.infinity,
            height: double.infinity,
            decoration: BoxDecoration(
              gradient: RadialGradient(
                center: Alignment.topLeft,
                radius: 2.5,
                colors: [
                  _colorTransitionAnimation.value ?? Color(0xFF0D47A1),
                  Color(0xFF1976D2),
                  Color(0xFF42A5F5),
                  Color.lerp(
                    Color(0xFF81C784),
                    Color(0xFF4FC3F7),
                    _morphAnimation.value,
                  )!,
                  Color(0xFF00BCD4).withOpacity(0.8),
                ],
                stops: [0.0, 0.15, 0.4, 0.7, 1.0],
              ),
            ),
            child: Stack(
              children: [
                // Morphing background shapes
                AnimatedBuilder(
                  animation: _morphController,
                  builder: (context, child) {
                    return CustomPaint(
                      painter: MorphingBackgroundPainter(
                        morphingShapes,
                        _morphAnimation.value,
                      ),
                      size: Size.infinite,
                    );
                  },
                ),

                // Enhanced particle system with trails
                AnimatedBuilder(
                  animation: _particleController,
                  builder: (context, child) {
                    return CustomPaint(
                      painter: EnhancedParticlePainter(
                        particles,
                        _particleController.value,
                      ),
                      size: Size.infinite,
                    );
                  },
                ),

                // Liquid bubble layer
                AnimatedBuilder(
                  animation: _liquidController,
                  builder: (context, child) {
                    return CustomPaint(
                      painter: LiquidBubblePainter(
                        bubbles,
                        _liquidWaveAnimation.value,
                      ),
                      size: Size.infinite,
                    );
                  },
                ),

                // Orbiting elements around logo
                AnimatedBuilder(
                  animation: _orbitController,
                  builder: (context, child) {
                    return CustomPaint(
                      painter: OrbitingElementsPainter(
                        orbitingElements,
                        _orbitAnimation.value,
                      ),
                      size: Size.infinite,
                    );
                  },
                ),

                // Main content
                Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      // Ultra-animated logo with multiple effects
                      AnimatedBuilder(
                        animation: Listenable.merge([
                          _scaleAnimation,
                          _rotationAnimation,
                          _logoBreathAnimation,
                          _glowAnimation,
                        ]),
                        builder: (context, child) {
                          return Transform.scale(
                            scale: _scaleAnimation.value *
                                _logoBreathAnimation.value,
                            child: Transform.rotate(
                              angle: _rotationAnimation.value * 0.15,
                              child: Container(
                                width: 180,
                                height: 180,
                                decoration: BoxDecoration(
                                  borderRadius: BorderRadius.circular(90),
                                  boxShadow: [
                                    // Multiple layered shadows for depth
                                    BoxShadow(
                                      color: Colors.black.withOpacity(0.4),
                                      blurRadius: 40,
                                      spreadRadius: 8,
                                      offset: Offset(0, 20),
                                    ),
                                    BoxShadow(
                                      color: Colors.blue.withOpacity(
                                        0.6 * _glowAnimation.value,
                                      ),
                                      blurRadius: 60,
                                      spreadRadius: -5,
                                      offset: Offset(0, 0),
                                    ),
                                    BoxShadow(
                                      color: Colors.cyan.withOpacity(
                                        0.4 * _glowAnimation.value,
                                      ),
                                      blurRadius: 80,
                                      spreadRadius: -10,
                                      offset: Offset(0, 0),
                                    ),
                                  ],
                                  gradient: RadialGradient(
                                    colors: [
                                      Colors.white,
                                      Colors.blue.shade50,
                                      Colors.blue.shade100,
                                    ],
                                  ),
                                ),
                                child: Stack(
                                  alignment: Alignment.center,
                                  children: [
                                    // Animated ripple effects
                                    ...List.generate(3, (index) {
                                      return AnimatedBuilder(
                                        animation: _waveController,
                                        builder: (context, child) {
                                          final rippleValue =
                                              (_waveController.value +
                                                      index * 0.3) %
                                                  1.0;
                                          return Transform.scale(
                                            scale: 0.5 + rippleValue * 0.8,
                                            child: Container(
                                              width: 180,
                                              height: 180,
                                              decoration: BoxDecoration(
                                                shape: BoxShape.circle,
                                                border: Border.all(
                                                  color:
                                                      Colors.white.withOpacity(
                                                    (1 - rippleValue) * 0.5,
                                                  ),
                                                  width: 3,
                                                ),
                                              ),
                                            ),
                                          );
                                        },
                                      );
                                    }),

                                    // Main water drop with enhanced shader
                                    ShaderMask(
                                      shaderCallback: (bounds) {
                                        return SweepGradient(
                                          center: Alignment.center,
                                          startAngle: 0,
                                          endAngle: 2 * pi,
                                          colors: [
                                            Colors.blue.shade700,
                                            Colors.blue.shade500,
                                            Colors.cyan.shade400,
                                            Colors.blue.shade600,
                                            Colors.blue.shade700,
                                          ],
                                        ).createShader(bounds);
                                      },
                                      child: Icon(
                                        Icons.water_drop,
                                        size: 100,
                                        color: Colors.white,
                                      ),
                                    ),

                                    // Enhanced microscope overlay
                                    Positioned(
                                      bottom: 30,
                                      right: 30,
                                      child: AnimatedBuilder(
                                        animation: _waveController,
                                        builder: (context, child) {
                                          return Transform.scale(
                                            scale: 1.0 +
                                                sin(
                                                      _waveController.value *
                                                          2 *
                                                          pi,
                                                    ) *
                                                    0.15,
                                            child: Container(
                                              padding: EdgeInsets.all(10),
                                              decoration: BoxDecoration(
                                                color: Colors.orange.shade400,
                                                borderRadius:
                                                    BorderRadius.circular(25),
                                                boxShadow: [
                                                  BoxShadow(
                                                    color: Colors.orange
                                                        .withOpacity(0.6),
                                                    blurRadius: 15,
                                                    spreadRadius: 3,
                                                  ),
                                                ],
                                              ),
                                              child: Icon(
                                                Icons.biotech,
                                                size: 28,
                                                color: Colors.white,
                                              ),
                                            ),
                                          );
                                        },
                                      ),
                                    ),

                                    // DNA helix overlay
                                    Positioned(
                                      top: 25,
                                      left: 25,
                                      child: AnimatedBuilder(
                                        animation: _rotationAnimation,
                                        builder: (context, child) {
                                          return Transform.rotate(
                                            angle: _rotationAnimation.value,
                                            child: Container(
                                              padding: EdgeInsets.all(8),
                                              decoration: BoxDecoration(
                                                color: Colors.green.shade400,
                                                borderRadius:
                                                    BorderRadius.circular(20),
                                                boxShadow: [
                                                  BoxShadow(
                                                    color: Colors.green
                                                        .withOpacity(0.5),
                                                    blurRadius: 12,
                                                    spreadRadius: 2,
                                                  ),
                                                ],
                                              ),
                                              child: Icon(
                                                Icons.science,
                                                size: 22,
                                                color: Colors.white,
                                              ),
                                            ),
                                          );
                                        },
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          );
                        },
                      ),

                      SizedBox(height: 60),

                      // Enhanced title with advanced animations
                      AnimatedBuilder(
                        animation: _textRevealAnimation,
                        builder: (context, child) {
                          return Transform.translate(
                            offset: Offset(0, _slideAnimation.value),
                            child: Opacity(
                              opacity: _textRevealAnimation.value,
                              child: Column(
                                children: [
                                  // Main title with complex shader
                                  ShaderMask(
                                    shaderCallback: (bounds) {
                                      return LinearGradient(
                                        colors: [
                                          Colors.white,
                                          Colors.blue.shade100,
                                          Colors.cyan.shade200,
                                          Colors.white,
                                        ],
                                        stops: [0.0, 0.3, 0.7, 1.0],
                                      ).createShader(bounds);
                                    },
                                    child: Text(
                                      'Microplastic',
                                      style: TextStyle(
                                        fontSize: 48,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.white,
                                        letterSpacing: 3.0,
                                        shadows: [
                                          Shadow(
                                            color: Colors.black.withOpacity(
                                              0.5,
                                            ),
                                            blurRadius: 15,
                                            offset: Offset(4, 4),
                                          ),
                                          Shadow(
                                            color: Colors.blue.withOpacity(0.3),
                                            blurRadius: 25,
                                            offset: Offset(-2, -2),
                                          ),
                                        ],
                                      ),
                                    ),
                                  ),

                                  // Subtitle with wave effect
                                  AnimatedBuilder(
                                    animation: _waveController,
                                    builder: (context, child) {
                                      return ShaderMask(
                                        shaderCallback: (bounds) {
                                          return LinearGradient(
                                            colors: [
                                              Colors.white.withOpacity(0.7),
                                              Colors.cyan.shade100,
                                              Colors.white.withOpacity(0.7),
                                            ],
                                            stops: [0.0, 0.5, 1.0],
                                          ).createShader(bounds);
                                        },
                                        child: Text(
                                          'Detector',
                                          style: TextStyle(
                                            fontSize: 48,
                                            fontWeight: FontWeight.w200,
                                            color: Colors.white,
                                            letterSpacing: 4.0,
                                            shadows: [
                                              Shadow(
                                                color: Colors.black.withOpacity(
                                                  0.4,
                                                ),
                                                blurRadius: 12,
                                                offset: Offset(3, 3),
                                              ),
                                            ],
                                          ),
                                        ),
                                      );
                                    },
                                  ),
                                ],
                              ),
                            ),
                          );
                        },
                      ),

                      SizedBox(height: 35),

                      // Enhanced subtitle with glassmorphism
                      FadeTransition(
                        opacity: _fadeAnimation,
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(25),
                          child: BackdropFilter(
                            filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                            child: Container(
                              padding: EdgeInsets.symmetric(
                                horizontal: 24,
                                vertical: 12,
                              ),
                              decoration: BoxDecoration(
                                color: Colors.white.withOpacity(0.15),
                                borderRadius: BorderRadius.circular(25),
                                border: Border.all(
                                  color: Colors.white.withOpacity(0.3),
                                  width: 1.5,
                                ),
                              ),
                              child: Text(
                                'Advanced Environmental Monitoring System',
                                style: TextStyle(
                                  fontSize: 18,
                                  color: Colors.white.withOpacity(0.95),
                                  letterSpacing: 1.2,
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ),
                          ),
                        ),
                      ),

                      SizedBox(height: 100),

                      // Ultra-sophisticated loading animation
                      AnimatedBuilder(
                        animation: Listenable.merge([
                          _loadingAnimation,
                          _glowAnimation,
                        ]),
                        builder: (context, child) {
                          return Column(
                            children: [
                              // Complex loading rings
                              Stack(
                                alignment: Alignment.center,
                                children: [
                                  // Outer pulsing ring
                                  Transform.scale(
                                    scale: 1.0 +
                                        sin(_loadingAnimation.value * 2 * pi) *
                                            0.4,
                                    child: Container(
                                      width: 100,
                                      height: 100,
                                      decoration: BoxDecoration(
                                        shape: BoxShape.circle,
                                        border: Border.all(
                                          color: Colors.white.withOpacity(0.2),
                                          width: 2,
                                        ),
                                      ),
                                    ),
                                  ),
                                  // Middle ring
                                  Transform.scale(
                                    scale: 1.0 +
                                        cos(
                                              _loadingAnimation.value * 2 * pi +
                                                  pi / 3,
                                            ) *
                                            0.25,
                                    child: Container(
                                      width: 75,
                                      height: 75,
                                      decoration: BoxDecoration(
                                        shape: BoxShape.circle,
                                        border: Border.all(
                                          color: Colors.white.withOpacity(0.4),
                                          width: 2,
                                        ),
                                      ),
                                    ),
                                  ),
                                  // Inner ring
                                  Transform.scale(
                                    scale: 1.0 +
                                        sin(
                                              _loadingAnimation.value * 2 * pi +
                                                  pi / 2,
                                            ) *
                                            0.15,
                                    child: Container(
                                      width: 50,
                                      height: 50,
                                      decoration: BoxDecoration(
                                        shape: BoxShape.circle,
                                        border: Border.all(
                                          color: Colors.white.withOpacity(0.6),
                                          width: 1.5,
                                        ),
                                      ),
                                    ),
                                  ),
                                  // Central spinning element
                                  Transform.rotate(
                                    angle: _loadingAnimation.value * 4 * pi,
                                    child: Container(
                                      width: 30,
                                      height: 30,
                                      decoration: BoxDecoration(
                                        shape: BoxShape.circle,
                                        gradient: SweepGradient(
                                          colors: [
                                            Colors.transparent,
                                            Colors.white.withOpacity(0.9),
                                            Colors.transparent,
                                          ],
                                        ),
                                      ),
                                    ),
                                  ),
                                  // DNA-like helix
                                  Transform.rotate(
                                    angle: -_loadingAnimation.value * 3 * pi,
                                    child: Container(
                                      width: 65,
                                      height: 65,
                                      decoration: BoxDecoration(
                                        shape: BoxShape.circle,
                                        gradient: SweepGradient(
                                          colors: [
                                            Colors.transparent,
                                            Colors.cyan.withOpacity(0.6),
                                            Colors.transparent,
                                            Colors.blue.withOpacity(0.4),
                                            Colors.transparent,
                                          ],
                                        ),
                                      ),
                                    ),
                                  ),
                                ],
                              ),

                              SizedBox(height: 40),

                              // Dynamic loading text with typewriter effect
                              AnimatedSwitcher(
                                duration: Duration(milliseconds: 300),
                                child: Container(
                                  key: ValueKey(currentLoadingIndex),
                                  child: Text(
                                    loadingTexts[currentLoadingIndex],
                                    style: TextStyle(
                                      fontSize: 18,
                                      color: Colors.white.withOpacity(0.9),
                                      letterSpacing: 1.0,
                                      fontWeight: FontWeight.w400,
                                    ),
                                  ),
                                ),
                              ),

                              SizedBox(height: 20),

                              // Progress indicator
                              Container(
                                width: 200,
                                height: 4,
                                decoration: BoxDecoration(
                                  color: Colors.white.withOpacity(0.2),
                                  borderRadius: BorderRadius.circular(2),
                                ),
                                child: AnimatedBuilder(
                                  animation: _loadingAnimation,
                                  builder: (context, child) {
                                    return FractionallySizedBox(
                                      widthFactor:
                                          (_loadingAnimation.value * 0.8 +
                                                  0.2) *
                                              (currentLoadingIndex + 1) /
                                              loadingTexts.length,
                                      child: Container(
                                        height: 4,
                                        decoration: BoxDecoration(
                                          gradient: LinearGradient(
                                            colors: [
                                              Colors.cyan.withOpacity(0.8),
                                              Colors.blue.withOpacity(0.9),
                                              Colors.white,
                                            ],
                                          ),
                                          borderRadius: BorderRadius.circular(
                                            2,
                                          ),
                                        ),
                                      ),
                                    );
                                  },
                                ),
                              ),
                            ],
                          );
                        },
                      ),
                    ],
                  ),
                ),

                // Enhanced bottom branding with animation
                Positioned(
                  bottom: 50,
                  left: 0,
                  right: 0,
                  child: FadeTransition(
                    opacity: _fadeAnimation,
                    child: Column(
                      children: [
                        ClipRRect(
                          borderRadius: BorderRadius.circular(20),
                          child: BackdropFilter(
                            filter: ImageFilter.blur(sigmaX: 8, sigmaY: 8),
                            child: Container(
                              padding: EdgeInsets.symmetric(
                                horizontal: 20,
                                vertical: 10,
                              ),
                              decoration: BoxDecoration(
                                color: Colors.white.withOpacity(0.1),
                                borderRadius: BorderRadius.circular(20),
                                border: Border.all(
                                  color: Colors.white.withOpacity(0.2),
                                ),
                              ),
                              child: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  AnimatedBuilder(
                                    animation: _waveController,
                                    builder: (context, child) {
                                      return Transform.rotate(
                                        angle: _waveController.value * 2 * pi,
                                        child: Icon(
                                          Icons.verified,
                                          size: 18,
                                          color: Colors.green.shade300,
                                        ),
                                      );
                                    },
                                  ),
                                  SizedBox(width: 10),
                                  Text(
                                    'v2.0.0 • Neural Engine Edition',
                                    style: TextStyle(
                                      fontSize: 14,
                                      color: Colors.white.withOpacity(0.85),
                                      fontWeight: FontWeight.w500,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ),
                        SizedBox(height: 12),
                        Text(
                          'Environmental Intelligence Initiative',
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.white.withOpacity(0.6),
                            letterSpacing: 1.5,
                            fontWeight: FontWeight.w300,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}

// Enhanced data classes for visual elements
class EnhancedParticle {
  double x, y, size, speed, oscillation, phase;
  Color color;
  int trailLength;
  List<Offset> trail = [];

  EnhancedParticle({
    required this.x,
    required this.y,
    required this.size,
    required this.speed,
    required this.color,
    required this.oscillation,
    required this.phase,
    required this.trailLength,
  });
}

class LiquidBubble {
  double x, y, size, speed, pulsePhase;
  Color color;

  LiquidBubble({
    required this.x,
    required this.y,
    required this.size,
    required this.speed,
    required this.color,
    required this.pulsePhase,
  });
}

class OrbitingElement {
  double centerX, centerY, radius, speed, size, phase;
  Color color;

  OrbitingElement({
    required this.centerX,
    required this.centerY,
    required this.radius,
    required this.speed,
    required this.size,
    required this.phase,
    required this.color,
  });
}

class MorphingShape {
  double x, y, size, speed, morphPhase;
  int shapeType;

  MorphingShape({
    required this.x,
    required this.y,
    required this.size,
    required this.speed,
    required this.morphPhase,
    required this.shapeType,
  });
}

// Custom painters for visual effects
class EnhancedParticlePainter extends CustomPainter {
  final List<EnhancedParticle> particles;
  final double animationValue;

  EnhancedParticlePainter(this.particles, this.animationValue);

  @override
  void paint(Canvas canvas, Size size) {
    for (var particle in particles) {
      final currentY = (particle.y + animationValue * particle.speed) % 1.0;
      final currentX = particle.x +
          sin(animationValue * 2 * pi * particle.oscillation + particle.phase) *
              0.08;

      final currentPos = Offset(currentX * size.width, currentY * size.height);

      // Update trail
      particle.trail.add(currentPos);
      if (particle.trail.length > particle.trailLength) {
        particle.trail.removeAt(0);
      }

      // Draw trail
      for (int i = 0; i < particle.trail.length - 1; i++) {
        final opacity = (i + 1) / particle.trail.length;
        final trailPaint = Paint()
          ..color = particle.color.withOpacity(opacity * 0.6)
          ..strokeWidth = particle.size * opacity
          ..strokeCap = StrokeCap.round;

        canvas.drawLine(particle.trail[i], particle.trail[i + 1], trailPaint);
      }

      // Draw main particle with glow
      final glowPaint = Paint()
        ..color = particle.color.withOpacity(0.3)
        ..maskFilter = MaskFilter.blur(BlurStyle.normal, particle.size);

      final mainPaint = Paint()
        ..color = particle.color
        ..style = PaintingStyle.fill;

      canvas.drawCircle(currentPos, particle.size * 2, glowPaint);
      canvas.drawCircle(currentPos, particle.size, mainPaint);
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}

class LiquidBubblePainter extends CustomPainter {
  final List<LiquidBubble> bubbles;
  final double animationValue;

  LiquidBubblePainter(this.bubbles, this.animationValue);

  @override
  void paint(Canvas canvas, Size size) {
    for (var bubble in bubbles) {
      final currentY = (bubble.y + animationValue * bubble.speed * 0.5) % 1.0;
      final currentX =
          bubble.x + sin(animationValue + bubble.pulsePhase) * 0.05;
      final currentSize = bubble.size +
          sin(animationValue * 3 + bubble.pulsePhase) * bubble.size * 0.2;

      final paint = Paint()
        ..color = bubble.color
        ..style = PaintingStyle.fill
        ..maskFilter = MaskFilter.blur(BlurStyle.normal, 8);

      canvas.drawCircle(
        Offset(currentX * size.width, currentY * size.height),
        currentSize,
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}

class OrbitingElementsPainter extends CustomPainter {
  final List<OrbitingElement> elements;
  final double animationValue;

  OrbitingElementsPainter(this.elements, this.animationValue);

  @override
  void paint(Canvas canvas, Size size) {
    for (var element in elements) {
      final angle = animationValue * element.speed + element.phase;
      final centerX = element.centerX * size.width;
      final centerY = element.centerY * size.height;

      final x = centerX + cos(angle) * element.radius;
      final y = centerY + sin(angle) * element.radius;

      final paint = Paint()
        ..color = element.color
        ..style = PaintingStyle.fill;

      final glowPaint = Paint()
        ..color = element.color.withOpacity(0.4)
        ..maskFilter = MaskFilter.blur(BlurStyle.normal, element.size);

      canvas.drawCircle(Offset(x, y), element.size * 1.5, glowPaint);
      canvas.drawCircle(Offset(x, y), element.size, paint);
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}

class MorphingBackgroundPainter extends CustomPainter {
  final List<MorphingShape> shapes;
  final double animationValue;

  MorphingBackgroundPainter(this.shapes, this.animationValue);

  @override
  void paint(Canvas canvas, Size size) {
    for (var shape in shapes) {
      final currentY = (shape.y + animationValue * shape.speed) % 1.0;
      final morphValue =
          sin(animationValue * 2 * pi + shape.morphPhase) * 0.5 + 0.5;
      final currentSize = shape.size + morphValue * shape.size * 0.3;

      final paint = Paint()
        ..color = Colors.white.withOpacity(0.02 + morphValue * 0.03)
        ..style = PaintingStyle.fill;

      final center = Offset(shape.x * size.width, currentY * size.height);

      switch (shape.shapeType) {
        case 0: // Circle
          canvas.drawCircle(center, currentSize, paint);
          break;
        case 1: // Rectangle
          final rect = Rect.fromCenter(
            center: center,
            width: currentSize * 2,
            height: currentSize,
          );
          canvas.drawRect(rect, paint);
          break;
        case 2: // Triangle
          final path = Path();
          path.moveTo(center.dx, center.dy - currentSize);
          path.lineTo(center.dx - currentSize, center.dy + currentSize);
          path.lineTo(center.dx + currentSize, center.dy + currentSize);
          path.close();
          canvas.drawPath(path, paint);
          break;
      }
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}

// Simple placeholder screens
class LoginScreen extends StatelessWidget {
  const LoginScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFF0D47A1), Color(0xFF42A5F5)],
          ),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.water_drop, size: 100, color: Colors.white),
              SizedBox(height: 20),
              Text(
                'Login Screen',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: () =>
                    Navigator.pushReplacementNamed(context, '/dashboard'),
                child: Text('Go to Dashboard'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Microplastic Detector'),
        backgroundColor: Colors.blue.shade600,
        foregroundColor: Colors.white,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.blue.shade50, Colors.white],
          ),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.dashboard, size: 100, color: Colors.blue.shade600),
              SizedBox(height: 20),
              Text(
                'Dashboard Screen',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Colors.blue.shade800,
                ),
              ),
              SizedBox(height: 20),
              Text(
                'Firebase connected successfully!',
                style: TextStyle(color: Colors.green.shade600),
              ),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  FirebaseAuth.instance.signOut();
                  Navigator.pushReplacementNamed(context, '/login');
                },
                child: Text('Sign Out'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
