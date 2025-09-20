import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:math';
import 'dart:async';
import 'dart:ui';

class EnhancedLoginScreen extends StatefulWidget {
  const EnhancedLoginScreen({super.key});

  @override
  _EnhancedLoginScreenState createState() => _EnhancedLoginScreenState();
}

class _EnhancedLoginScreenState extends State<EnhancedLoginScreen>
    with TickerProviderStateMixin {
  // Animation Controllers
  late AnimationController _backgroundController;
  late AnimationController _formController;
  late AnimationController _logoController;
  late AnimationController _particleController;
  late AnimationController _buttonController;
  late AnimationController _fieldController;
  late AnimationController _modeTransitionController;

  // Animations
  late Animation<double> _backgroundAnimation;
  late Animation<double> _formSlideAnimation;
  late Animation<double> _formFadeAnimation;
  late Animation<double> _logoFloatAnimation;
  late Animation<double> _logoScaleAnimation;
  late Animation<double> _buttonScaleAnimation;
  late Animation<double> _fieldFocusAnimation;
  late Animation<double> _modeTransitionAnimation;
  late Animation<Color?> _colorTransitionAnimation;

  // Form Controllers
  final _usernameController = TextEditingController();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();
  final _formKey = GlobalKey<FormState>();

  // Focus Nodes
  final _usernameFocus = FocusNode();
  final _emailFocus = FocusNode();
  final _passwordFocus = FocusNode();
  final _confirmPasswordFocus = FocusNode();

  // State Variables
  bool _isLoginMode = true;
  bool _isLoading = false;
  bool _obscurePassword = true;
  bool _obscureConfirmPassword = true;
  bool _rememberMe = false;
  String _errorMessage = '';
  String _successMessage = '';

  // Firebase instances
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final GoogleSignIn _googleSignIn = GoogleSignIn();

  // Visual Elements
  List<CreativeParticle> particles = [];
  List<FloatingElement> floatingElements = [];

  @override
  void initState() {
    super.initState();
    _initializeAnimations();
    _generateVisualElements();
    _startAnimations();
    _setupFocusListeners();
    _loadRememberMeState();
  }

  void _initializeAnimations() {
    // Controllers
    _backgroundController = AnimationController(
      duration: Duration(seconds: 20),
      vsync: this,
    )..repeat();

    _formController = AnimationController(
      duration: Duration(milliseconds: 1500),
      vsync: this,
    );

    _logoController = AnimationController(
      duration: Duration(seconds: 3),
      vsync: this,
    )..repeat(reverse: true);

    _particleController = AnimationController(
      duration: Duration(seconds: 15),
      vsync: this,
    )..repeat();

    _buttonController = AnimationController(
      duration: Duration(milliseconds: 150),
      vsync: this,
    );

    _fieldController = AnimationController(
      duration: Duration(milliseconds: 200),
      vsync: this,
    );

    _modeTransitionController = AnimationController(
      duration: Duration(milliseconds: 600),
      vsync: this,
    );

    // Animations
    _backgroundAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(_backgroundController);

    _formSlideAnimation = Tween<double>(begin: 50.0, end: 0.0).animate(
      CurvedAnimation(parent: _formController, curve: Curves.easeOutCubic),
    );

    _formFadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _formController, curve: Curves.easeInOut),
    );

    _logoFloatAnimation = Tween<double>(begin: -10.0, end: 10.0).animate(
      CurvedAnimation(parent: _logoController, curve: Curves.easeInOutSine),
    );

    _logoScaleAnimation = Tween<double>(begin: 0.8, end: 1.0).animate(
      CurvedAnimation(parent: _formController, curve: Curves.elasticOut),
    );

    _buttonScaleAnimation = Tween<double>(begin: 1.0, end: 0.95).animate(
      CurvedAnimation(parent: _buttonController, curve: Curves.easeInOut),
    );

    _fieldFocusAnimation = Tween<double>(begin: 1.0, end: 1.02).animate(
      CurvedAnimation(parent: _fieldController, curve: Curves.easeInOut),
    );

    _modeTransitionAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _modeTransitionController,
        curve: Curves.easeInOut,
      ),
    );

    _colorTransitionAnimation = ColorTween(
      begin: Color(0xFF0D47A1),
      end: Color(0xFF6A1B9A),
    ).animate(_backgroundController);
  }

  void _generateVisualElements() {
    final random = Random();

    // Generate particles
    particles.clear();
    for (int i = 0; i < 30; i++) {
      particles.add(
        CreativeParticle(
          x: random.nextDouble(),
          y: random.nextDouble(),
          size: random.nextDouble() * 6 + 2,
          speed: random.nextDouble() * 0.8 + 0.2,
          color: Color.lerp(
            Colors.blue.withOpacity(0.3),
            Colors.purple.withOpacity(0.5),
            random.nextDouble(),
          )!,
          oscillation: random.nextDouble() * 3 + 1,
          phase: random.nextDouble() * 2 * pi,
        ),
      );
    }

    // Generate floating elements
    List<IconData> iconList = [
      Icons.water_drop,
      Icons.science,
      Icons.biotech,
      Icons.eco,
      Icons.nature,
      Icons.analytics,
      Icons.security,
      Icons.verified_user,
    ];

    floatingElements.clear();
    for (int i = 0; i < 8; i++) {
      floatingElements.add(
        FloatingElement(
          x: random.nextDouble(),
          y: random.nextDouble(),
          icon: iconList[i],
          size: random.nextDouble() * 12 + 8,
          speed: random.nextDouble() * 0.3 + 0.1,
          rotationSpeed: random.nextDouble() * 0.5 + 0.2,
          opacity: random.nextDouble() * 0.15 + 0.05,
          glowIntensity: random.nextDouble() * 0.3 + 0.1,
        ),
      );
    }
  }

  void _startAnimations() {
    Future.delayed(Duration(milliseconds: 300), () {
      if (mounted) {
        _formController.forward();
      }
    });
  }

  void _setupFocusListeners() {
    for (var focus in [
      _usernameFocus,
      _emailFocus,
      _passwordFocus,
      _confirmPasswordFocus,
    ]) {
      focus.addListener(() {
        if (mounted) {
          if (focus.hasFocus) {
            _fieldController.forward();
          } else {
            _fieldController.reverse();
          }
        }
      });
    }
  }

  Future<void> _loadRememberMeState() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final rememberedUsername = prefs.getString('remembered_username');
      final rememberMe = prefs.getBool('remember_me') ?? false;

      if (mounted && rememberedUsername != null && rememberMe) {
        setState(() {
          _rememberMe = true;
          _usernameController.text = rememberedUsername;
        });
      }
    } catch (e) {
      print('Error loading remember me state: $e');
    }
  }

  Future<void> _saveRememberMeState() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      if (_rememberMe && _usernameController.text.isNotEmpty) {
        await prefs.setString(
          'remembered_username',
          _usernameController.text.trim(),
        );
        await prefs.setBool('remember_me', true);
      } else {
        await prefs.remove('remembered_username');
        await prefs.setBool('remember_me', false);
      }
    } catch (e) {
      print('Error saving remember me state: $e');
    }
  }

  Future<void> _handleAuthentication() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _isLoading = true;
      _errorMessage = '';
      _successMessage = '';
    });

    _buttonController.forward();

    try {
      if (_isLoginMode) {
        await _handleLogin();
      } else {
        await _handleRegister();
      }

      await _saveRememberMeState();
    } catch (e) {
      setState(() {
        _errorMessage = e.toString();
      });
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
        _buttonController.reverse();
      }
    }
  }

  Future<void> _handleLogin() async {
    try {
      // First, try to find user by username in Firestore
      final usernameQuery = await _firestore
          .collection('users')
          .where('username', isEqualTo: _usernameController.text.trim())
          .get();

      if (usernameQuery.docs.isEmpty) {
        throw FirebaseAuthException(
          code: 'user-not-found',
          message: 'Username not found. Would you like to create an account?',
        );
      }

      // Get the email associated with the username
      final userData = usernameQuery.docs.first.data();
      final email = userData['email'] as String?;

      if (email == null) {
        throw FirebaseAuthException(
          code: 'invalid-user-data',
          message: 'User data is corrupted. Please contact support.',
        );
      }

      // Sign in with email and password
      await _auth.signInWithEmailAndPassword(
        email: email,
        password: _passwordController.text,
      );

      _showSuccessMessage('Login successful! Welcome back.');

      // Navigate to dashboard
      if (Navigator.canPop(context)) {
        Navigator.pushReplacementNamed(context, '/dashboard');
      }
    } catch (e) {
      rethrow;
    }
  }

  Future<void> _handleRegister() async {
    try {
      // Check if username already exists
      final usernameQuery = await _firestore
          .collection('users')
          .where('username', isEqualTo: _usernameController.text.trim())
          .get();

      if (usernameQuery.docs.isNotEmpty) {
        throw FirebaseAuthException(
          code: 'username-already-exists',
          message:
              'Username already exists. Please choose a different username.',
        );
      }

      // Use the provided email
      String email = _emailController.text.trim();

      // Create user account with Firebase Auth
      UserCredential userCredential =
          await _auth.createUserWithEmailAndPassword(
        email: email,
        password: _passwordController.text,
      );

      // Check if user creation was successful
      if (userCredential.user == null) {
        throw FirebaseAuthException(
          code: 'user-creation-failed',
          message: 'Failed to create user account.',
        );
      }

      // Save user data to Firestore
      await _firestore.collection('users').doc(userCredential.user!.uid).set({
        'uid': userCredential.user!.uid,
        'username': _usernameController.text.trim(),
        'email': email,
        'displayName': _usernameController.text.trim(),
        'createdAt': FieldValue.serverTimestamp(),
        'isEmailVerified': false,
      });

      // Update user display name
      await userCredential.user?.updateDisplayName(
        _usernameController.text.trim(),
      );

      _showSuccessMessage(
        'Account created successfully! Welcome to Microplastic Detector.',
      );

      // Navigate to dashboard
      if (Navigator.canPop(context)) {
        Navigator.pushReplacementNamed(context, '/dashboard');
      }
    } catch (e) {
      rethrow;
    }
  }

  Future<String> _generateUniqueUsername(String baseName) async {
    String username = baseName;
    int counter = 1;

    while (true) {
      final query = await _firestore
          .collection('users')
          .where('username', isEqualTo: username)
          .get();

      if (query.docs.isEmpty) {
        return username;
      }

      username = '$baseName$counter';
      counter++;
    }
  }

  Future<void> _handleGoogleSignIn() async {
    setState(() {
      _isLoading = true;
      _errorMessage = '';
    });

    try {
      final GoogleSignInAccount? googleUser = await _googleSignIn.signIn();
      if (googleUser == null) {
        setState(() => _isLoading = false);
        return;
      }

      final GoogleSignInAuthentication googleAuth = googleUser.authentication;
      final credential = GoogleAuthProvider.credential(
        accessToken: googleAuth.accessToken,
        idToken: googleAuth.idToken,
      );

      UserCredential userCredential = await _auth.signInWithCredential(
        credential,
      );

      if (userCredential.user == null) {
        throw FirebaseAuthException(
          code: 'google-signin-failed',
          message: 'Google sign-in failed.',
        );
      }

      // Generate unique username from display name
      String baseUsername = userCredential.user!.displayName
              ?.replaceAll(RegExp(r'[^a-zA-Z0-9]'), '')
              .toLowerCase() ??
          userCredential.user!.email!
              .split('@')[0]
              .replaceAll(RegExp(r'[^a-zA-Z0-9]'), '');

      String uniqueUsername = await _generateUniqueUsername(baseUsername);

      // Save/update user data in Firestore
      await _firestore.collection('users').doc(userCredential.user!.uid).set({
        'uid': userCredential.user!.uid,
        'email': userCredential.user!.email,
        'username': uniqueUsername,
        'displayName': userCredential.user!.displayName,
        'photoURL': userCredential.user!.photoURL,
        'createdAt': FieldValue.serverTimestamp(),
        'isEmailVerified': true,
        'loginProvider': 'google',
      }, SetOptions(merge: true));

      _showSuccessMessage('Google sign-in successful!');

      if (Navigator.canPop(context)) {
        Navigator.pushReplacementNamed(context, '/dashboard');
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Google sign-in failed. Please try again.';
      });
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  void _showSuccessMessage(String message) {
    if (mounted) {
      setState(() {
        _successMessage = message;
        _errorMessage = '';
      });
      Timer(Duration(seconds: 3), () {
        if (mounted) {
          setState(() => _successMessage = '');
        }
      });
    }
  }

  void _switchMode() {
    _modeTransitionController.forward().then((_) {
      if (mounted) {
        setState(() {
          _isLoginMode = !_isLoginMode;
          _errorMessage = '';
          _successMessage = '';
        });
        _modeTransitionController.reverse();
      }
    });
  }

  String _getErrorMessage(String error) {
    if (error.contains('weak-password')) {
      return 'Password is too weak. Use at least 6 characters.';
    } else if (error.contains('email-already-in-use')) {
      return 'An account with this email already exists.';
    } else if (error.contains('wrong-password')) {
      return 'Incorrect password. Please try again.';
    } else if (error.contains('too-many-requests')) {
      return 'Too many attempts. Please try again later.';
    } else if (error.contains('invalid-email')) {
      return 'Please enter a valid email address.';
    } else if (error.contains('user-not-found')) {
      return 'Username not found. Would you like to create an account?';
    } else if (error.contains('username-already-exists')) {
      return 'Username already exists. Please choose a different username.';
    }
    return error
        .replaceAll('FirebaseAuthException: ', '')
        .replaceAll('[firebase_auth/user-not-found]', '');
  }

  @override
  void dispose() {
    _backgroundController.dispose();
    _formController.dispose();
    _logoController.dispose();
    _particleController.dispose();
    _buttonController.dispose();
    _fieldController.dispose();
    _modeTransitionController.dispose();

    _usernameController.dispose();
    _emailController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();

    _usernameFocus.dispose();
    _emailFocus.dispose();
    _passwordFocus.dispose();
    _confirmPasswordFocus.dispose();

    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: AnimatedBuilder(
        animation: _colorTransitionAnimation,
        builder: (context, child) {
          return Container(
            width: double.infinity,
            height: double.infinity,
            decoration: BoxDecoration(
              gradient: RadialGradient(
                center: Alignment.topLeft,
                radius: 2.2,
                colors: [
                  _colorTransitionAnimation.value ?? Color(0xFF0D47A1),
                  Color(0xFF1976D2),
                  Color(0xFF42A5F5),
                  Color(0xFF6A1B9A).withOpacity(0.8),
                  Color(0xFF9C27B0).withOpacity(0.6),
                ],
                stops: [0.0, 0.2, 0.5, 0.8, 1.0],
              ),
            ),
            child: Stack(
              children: [
                // Animated background particles
                AnimatedBuilder(
                  animation: _particleController,
                  builder: (context, child) {
                    return RepaintBoundary(
                      child: CustomPaint(
                        painter: CreativeParticlePainter(
                          particles,
                          _backgroundAnimation.value,
                        ),
                        size: Size.infinite,
                      ),
                    );
                  },
                ),

                // Floating elements
                AnimatedBuilder(
                  animation: _backgroundController,
                  builder: (context, child) {
                    return RepaintBoundary(
                      child: CustomPaint(
                        painter: FloatingElementPainter(
                          floatingElements,
                          _backgroundAnimation.value,
                        ),
                        size: Size.infinite,
                      ),
                    );
                  },
                ),

                SafeArea(
                  child: SingleChildScrollView(
                    physics: BouncingScrollPhysics(),
                    child: Padding(
                      padding: EdgeInsets.all(24.0),
                      child: Column(
                        children: [
                          SizedBox(height: 40),

                          // Animated logo with floating effect
                          AnimatedBuilder(
                            animation: Listenable.merge([
                              _logoFloatAnimation,
                              _logoScaleAnimation,
                            ]),
                            builder: (context, child) {
                              return Transform.translate(
                                offset: Offset(0, _logoFloatAnimation.value),
                                child: Transform.scale(
                                  scale: _logoScaleAnimation.value,
                                  child: Hero(
                                    tag: 'app_logo',
                                    child: Container(
                                      width: 140,
                                      height: 140,
                                      decoration: BoxDecoration(
                                        gradient: RadialGradient(
                                          colors: [
                                            Colors.white,
                                            Colors.blue.shade50,
                                            Colors.blue.shade100,
                                          ],
                                        ),
                                        borderRadius: BorderRadius.circular(70),
                                        boxShadow: [
                                          BoxShadow(
                                            color: Colors.black.withOpacity(
                                              0.3,
                                            ),
                                            blurRadius: 30,
                                            spreadRadius: 8,
                                            offset: Offset(0, 15),
                                          ),
                                          BoxShadow(
                                            color: Colors.blue.withOpacity(0.4),
                                            blurRadius: 40,
                                            spreadRadius: -5,
                                          ),
                                        ],
                                      ),
                                      child: Stack(
                                        alignment: Alignment.center,
                                        children: [
                                          Icon(
                                            Icons.water_drop,
                                            size: 70,
                                            color: Colors.blue.shade700,
                                          ),
                                          Positioned(
                                            bottom: 35,
                                            right: 35,
                                            child: Container(
                                              padding: EdgeInsets.all(8),
                                              decoration: BoxDecoration(
                                                color: Colors.orange.shade400,
                                                borderRadius:
                                                    BorderRadius.circular(20),
                                                boxShadow: [
                                                  BoxShadow(
                                                    color: Colors.orange
                                                        .withOpacity(0.5),
                                                    blurRadius: 10,
                                                    spreadRadius: 2,
                                                  ),
                                                ],
                                              ),
                                              child: Icon(
                                                Icons.biotech,
                                                size: 20,
                                                color: Colors.white,
                                              ),
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                  ),
                                ),
                              );
                            },
                          ),

                          SizedBox(height: 30),

                          // App title
                          AnimatedBuilder(
                            animation: _formFadeAnimation,
                            builder: (context, child) {
                              return Opacity(
                                opacity: _formFadeAnimation.value,
                                child: Column(
                                  children: [
                                    ShaderMask(
                                      shaderCallback: (bounds) {
                                        return LinearGradient(
                                          colors: [
                                            Colors.white,
                                            Colors.blue.shade100,
                                          ],
                                        ).createShader(bounds);
                                      },
                                      child: Text(
                                        'Microplastic Detector',
                                        style: TextStyle(
                                          fontSize: 32,
                                          fontWeight: FontWeight.bold,
                                          color: Colors.white,
                                          letterSpacing: 1.5,
                                        ),
                                      ),
                                    ),
                                    SizedBox(height: 8),
                                    Text(
                                      'Environmental Intelligence Platform',
                                      style: TextStyle(
                                        fontSize: 14,
                                        color: Colors.white.withOpacity(0.8),
                                        letterSpacing: 0.8,
                                      ),
                                    ),
                                  ],
                                ),
                              );
                            },
                          ),

                          SizedBox(height: 50),

                          // Main form card
                          AnimatedBuilder(
                            animation: Listenable.merge([
                              _formSlideAnimation,
                              _formFadeAnimation,
                              _modeTransitionAnimation,
                            ]),
                            builder: (context, child) {
                              return Transform.translate(
                                offset: Offset(0, _formSlideAnimation.value),
                                child: Opacity(
                                  opacity: _formFadeAnimation.value *
                                      (1 -
                                          _modeTransitionAnimation.value * 0.3),
                                  child: ClipRRect(
                                    borderRadius: BorderRadius.circular(28),
                                    child: BackdropFilter(
                                      filter: ImageFilter.blur(
                                        sigmaX: 15,
                                        sigmaY: 15,
                                      ),
                                      child: Container(
                                        padding: EdgeInsets.all(32),
                                        decoration: BoxDecoration(
                                          color: Colors.white.withOpacity(0.15),
                                          borderRadius: BorderRadius.circular(
                                            28,
                                          ),
                                          border: Border.all(
                                            color: Colors.white.withOpacity(
                                              0.3,
                                            ),
                                            width: 1.5,
                                          ),
                                          boxShadow: [
                                            BoxShadow(
                                              color: Colors.black.withOpacity(
                                                0.1,
                                              ),
                                              blurRadius: 40,
                                              spreadRadius: 5,
                                            ),
                                          ],
                                        ),
                                        child: Form(
                                          key: _formKey,
                                          child: Column(
                                            children: [
                                              // Mode selector
                                              _buildModeSelector(),

                                              SizedBox(height: 30),

                                              // Username field
                                              _buildAnimatedTextField(
                                                controller: _usernameController,
                                                focusNode: _usernameFocus,
                                                label: 'Username',
                                                icon: Icons.person,
                                                validator: (value) {
                                                  if (value?.isEmpty ?? true) {
                                                    return 'Username is required';
                                                  }
                                                  if (value!.length < 3) {
                                                    return 'Username must be at least 3 characters';
                                                  }
                                                  if (!RegExp(
                                                    r'^[a-zA-Z0-9_]+$',
                                                  ).hasMatch(value)) {
                                                    return 'Username can only contain letters, numbers, and underscores';
                                                  }
                                                  return null;
                                                },
                                              ),

                                              SizedBox(height: 20),

                                              // Email field (only for register mode)
                                              if (!_isLoginMode) ...[
                                                _buildAnimatedTextField(
                                                  controller: _emailController,
                                                  focusNode: _emailFocus,
                                                  label: 'Email',
                                                  icon: Icons.email,
                                                  validator: (value) {
                                                    if (value?.isEmpty ??
                                                        true) {
                                                      return 'Email is required';
                                                    }
                                                    if (!RegExp(
                                                      r'^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$',
                                                    ).hasMatch(value!)) {
                                                      return 'Please enter a valid email address';
                                                    }
                                                    return null;
                                                  },
                                                ),
                                                SizedBox(height: 20),
                                              ],

                                              // Password field
                                              _buildAnimatedTextField(
                                                controller: _passwordController,
                                                focusNode: _passwordFocus,
                                                label: 'Password',
                                                icon: Icons.lock,
                                                obscureText: _obscurePassword,
                                                suffixIcon: IconButton(
                                                  icon: Icon(
                                                    _obscurePassword
                                                        ? Icons.visibility
                                                        : Icons.visibility_off,
                                                    color: Colors.white
                                                        .withOpacity(0.7),
                                                  ),
                                                  onPressed: () {
                                                    setState(() {
                                                      _obscurePassword =
                                                          !_obscurePassword;
                                                    });
                                                  },
                                                ),
                                                validator: (value) {
                                                  if (value?.isEmpty ?? true) {
                                                    return 'Password is required';
                                                  }
                                                  if (!_isLoginMode &&
                                                      value!.length < 6) {
                                                    return 'Password must be at least 6 characters';
                                                  }
                                                  return null;
                                                },
                                              ),

                                              SizedBox(height: 20),

                                              // Confirm password field (only for register mode)
                                              if (!_isLoginMode) ...[
                                                _buildAnimatedTextField(
                                                  controller:
                                                      _confirmPasswordController,
                                                  focusNode:
                                                      _confirmPasswordFocus,
                                                  label: 'Confirm Password',
                                                  icon: Icons.lock_outline,
                                                  obscureText:
                                                      _obscureConfirmPassword,
                                                  suffixIcon: IconButton(
                                                    icon: Icon(
                                                      _obscureConfirmPassword
                                                          ? Icons.visibility
                                                          : Icons
                                                              .visibility_off,
                                                      color: Colors.white
                                                          .withOpacity(0.7),
                                                    ),
                                                    onPressed: () {
                                                      setState(() {
                                                        _obscureConfirmPassword =
                                                            !_obscureConfirmPassword;
                                                      });
                                                    },
                                                  ),
                                                  validator: (value) {
                                                    if (value?.isEmpty ??
                                                        true) {
                                                      return 'Please confirm your password';
                                                    }
                                                    if (value !=
                                                        _passwordController
                                                            .text) {
                                                      return 'Passwords do not match';
                                                    }
                                                    return null;
                                                  },
                                                ),
                                                SizedBox(height: 20),
                                              ],

                                              // Remember me (only for login mode)
                                              if (_isLoginMode) ...[
                                                _buildRememberMeCheckbox(),
                                                SizedBox(height: 20),
                                              ],

                                              // Error/Success messages
                                              _buildMessages(),

                                              SizedBox(height: 20),

                                              // Main action button
                                              _buildMainActionButton(),

                                              SizedBox(height: 25),

                                              // Divider
                                              _buildDivider(),

                                              SizedBox(height: 25),

                                              // Google sign in
                                              _buildGoogleSignInButton(),
                                            ],
                                          ),
                                        ),
                                      ),
                                    ),
                                  ),
                                ),
                              );
                            },
                          ),

                          SizedBox(height: 30),

                          // Mode switcher
                          _buildModeSwitcher(),
                        ],
                      ),
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

  Widget _buildModeSelector() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.1),
        borderRadius: BorderRadius.circular(25),
        border: Border.all(color: Colors.white.withOpacity(0.2)),
      ),
      child: Row(
        children: [
          Expanded(
            child: GestureDetector(
              onTap: () {
                if (!_isLoginMode) {
                  _switchMode();
                }
              },
              child: AnimatedContainer(
                duration: Duration(milliseconds: 300),
                padding: EdgeInsets.symmetric(vertical: 12),
                decoration: BoxDecoration(
                  color: _isLoginMode
                      ? Colors.white.withOpacity(0.25)
                      : Colors.transparent,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  'Sign In',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    color: _isLoginMode
                        ? Colors.white
                        : Colors.white.withOpacity(0.7),
                    fontWeight:
                        _isLoginMode ? FontWeight.bold : FontWeight.normal,
                    fontSize: 16,
                  ),
                ),
              ),
            ),
          ),
          Expanded(
            child: GestureDetector(
              onTap: () {
                if (_isLoginMode) {
                  _switchMode();
                }
              },
              child: AnimatedContainer(
                duration: Duration(milliseconds: 300),
                padding: EdgeInsets.symmetric(vertical: 12),
                decoration: BoxDecoration(
                  color: !_isLoginMode
                      ? Colors.white.withOpacity(0.25)
                      : Colors.transparent,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  'Register',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    color: !_isLoginMode
                        ? Colors.white
                        : Colors.white.withOpacity(0.7),
                    fontWeight:
                        !_isLoginMode ? FontWeight.bold : FontWeight.normal,
                    fontSize: 16,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAnimatedTextField({
    required TextEditingController controller,
    required FocusNode focusNode,
    required String label,
    required IconData icon,
    bool obscureText = false,
    Widget? suffixIcon,
    String? Function(String?)? validator,
  }) {
    return AnimatedBuilder(
      animation: _fieldFocusAnimation,
      builder: (context, child) {
        return Transform.scale(
          scale: focusNode.hasFocus ? _fieldFocusAnimation.value : 1.0,
          child: Container(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(18),
              boxShadow: focusNode.hasFocus
                  ? [
                      BoxShadow(
                        color: Colors.white.withOpacity(0.1),
                        blurRadius: 15,
                        spreadRadius: 3,
                      ),
                    ]
                  : null,
            ),
            child: TextFormField(
              controller: controller,
              focusNode: focusNode,
              obscureText: obscureText,
              validator: validator,
              style: TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.w500,
              ),
              decoration: InputDecoration(
                labelText: label,
                labelStyle: TextStyle(
                  color: Colors.white.withOpacity(
                    focusNode.hasFocus ? 1.0 : 0.7,
                  ),
                  fontSize: focusNode.hasFocus ? 14 : 16,
                  fontWeight: FontWeight.w400,
                ),
                prefixIcon: Icon(
                  icon,
                  color: Colors.white.withOpacity(
                    focusNode.hasFocus ? 1.0 : 0.7,
                  ),
                  size: 22,
                ),
                suffixIcon: suffixIcon,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(18),
                  borderSide: BorderSide(color: Colors.white.withOpacity(0.3)),
                ),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(18),
                  borderSide: BorderSide(
                    color: Colors.white.withOpacity(0.3),
                    width: 1.5,
                  ),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(18),
                  borderSide: BorderSide(color: Colors.white, width: 2.5),
                ),
                errorBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(18),
                  borderSide: BorderSide(
                    color: Colors.red.withOpacity(0.7),
                    width: 2,
                  ),
                ),
                focusedErrorBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(18),
                  borderSide: BorderSide(color: Colors.red, width: 2.5),
                ),
                filled: true,
                fillColor: Colors.white.withOpacity(
                  focusNode.hasFocus ? 0.18 : 0.1,
                ),
                contentPadding: EdgeInsets.symmetric(
                  vertical: 18,
                  horizontal: 20,
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildRememberMeCheckbox() {
    return Row(
      children: [
        Transform.scale(
          scale: 0.9,
          child: Checkbox(
            value: _rememberMe,
            onChanged: (value) {
              setState(() {
                _rememberMe = value ?? false;
              });
            },
            activeColor: Colors.white,
            checkColor: Colors.blue.shade700,
            side: BorderSide(color: Colors.white.withOpacity(0.7), width: 1.5),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(4),
            ),
          ),
        ),
        SizedBox(width: 8),
        Text(
          'Remember me',
          style: TextStyle(
            color: Colors.white.withOpacity(0.85),
            fontSize: 14,
            fontWeight: FontWeight.w400,
          ),
        ),
      ],
    );
  }

  Widget _buildMessages() {
    return AnimatedSwitcher(
      duration: Duration(milliseconds: 400),
      child: Column(
        key: ValueKey('${_errorMessage}_$_successMessage'),
        children: [
          if (_errorMessage.isNotEmpty)
            Container(
              margin: EdgeInsets.only(bottom: 12),
              padding: EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.red.withOpacity(0.15),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.red.withOpacity(0.4)),
              ),
              child: Row(
                children: [
                  Icon(
                    Icons.error_outline,
                    color: Colors.red.shade300,
                    size: 22,
                  ),
                  SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      _getErrorMessage(_errorMessage),
                      style: TextStyle(
                        color: Colors.red.shade300,
                        fontSize: 14,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          if (_successMessage.isNotEmpty)
            Container(
              margin: EdgeInsets.only(bottom: 12),
              padding: EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.green.withOpacity(0.15),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.green.withOpacity(0.4)),
              ),
              child: Row(
                children: [
                  Icon(
                    Icons.check_circle_outline,
                    color: Colors.green.shade300,
                    size: 22,
                  ),
                  SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      _successMessage,
                      style: TextStyle(
                        color: Colors.green.shade300,
                        fontSize: 14,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildMainActionButton() {
    return AnimatedBuilder(
      animation: _buttonScaleAnimation,
      builder: (context, child) {
        return Transform.scale(
          scale: _buttonScaleAnimation.value,
          child: Container(
            width: double.infinity,
            height: 60,
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  Colors.white.withOpacity(0.95),
                  Colors.white.withOpacity(0.8),
                ],
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
              ),
              borderRadius: BorderRadius.circular(30),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.2),
                  blurRadius: 20,
                  spreadRadius: 2,
                  offset: Offset(0, 8),
                ),
                BoxShadow(
                  color: Colors.white.withOpacity(0.1),
                  blurRadius: 10,
                  spreadRadius: -5,
                  offset: Offset(0, -2),
                ),
              ],
            ),
            child: ElevatedButton(
              onPressed: _isLoading ? null : _handleAuthentication,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.transparent,
                shadowColor: Colors.transparent,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(30),
                ),
              ),
              child: _isLoading
                  ? Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        SizedBox(
                          width: 24,
                          height: 24,
                          child: CircularProgressIndicator(
                            strokeWidth: 3,
                            valueColor: AlwaysStoppedAnimation<Color>(
                              Colors.blue.shade700,
                            ),
                          ),
                        ),
                        SizedBox(width: 12),
                        Text(
                          'Processing...',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                            color: Colors.blue.shade700,
                          ),
                        ),
                      ],
                    )
                  : Text(
                      _isLoginMode ? 'Sign In' : 'Create Account',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Colors.blue.shade700,
                        letterSpacing: 0.5,
                      ),
                    ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildDivider() {
    return Row(
      children: [
        Expanded(
          child: Container(
            height: 1.5,
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  Colors.transparent,
                  Colors.white.withOpacity(0.3),
                  Colors.white.withOpacity(0.3),
                  Colors.transparent,
                ],
              ),
            ),
          ),
        ),
        Padding(
          padding: EdgeInsets.symmetric(horizontal: 20),
          child: Container(
            padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.1),
              borderRadius: BorderRadius.circular(15),
              border: Border.all(color: Colors.white.withOpacity(0.2)),
            ),
            child: Text(
              'OR',
              style: TextStyle(
                color: Colors.white.withOpacity(0.8),
                fontSize: 12,
                fontWeight: FontWeight.w600,
                letterSpacing: 1,
              ),
            ),
          ),
        ),
        Expanded(
          child: Container(
            height: 1.5,
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  Colors.transparent,
                  Colors.white.withOpacity(0.3),
                  Colors.white.withOpacity(0.3),
                  Colors.transparent,
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildGoogleSignInButton() {
    return SizedBox(
      width: double.infinity,
      height: 56,
      child: OutlinedButton(
        onPressed: _isLoading ? null : _handleGoogleSignIn,
        style: OutlinedButton.styleFrom(
          foregroundColor: Colors.white,
          side: BorderSide(color: Colors.white.withOpacity(0.4), width: 2),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(28),
          ),
          backgroundColor: Colors.white.withOpacity(0.08),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: EdgeInsets.all(4),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Icon(Icons.g_mobiledata, color: Colors.red, size: 24),
            ),
            SizedBox(width: 12),
            Text(
              'Continue with Google',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w600,
                letterSpacing: 0.3,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildModeSwitcher() {
    return AnimatedBuilder(
      animation: _formFadeAnimation,
      builder: (context, child) {
        return Opacity(
          opacity: _formFadeAnimation.value,
          child: Container(
            padding: EdgeInsets.symmetric(horizontal: 20, vertical: 12),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.08),
              borderRadius: BorderRadius.circular(25),
              border: Border.all(color: Colors.white.withOpacity(0.15)),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  _isLoginMode
                      ? "Don't have an account? "
                      : "Already have an account? ",
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.8),
                    fontSize: 15,
                    fontWeight: FontWeight.w400,
                  ),
                ),
                GestureDetector(
                  onTap: _switchMode,
                  child: Container(
                    padding: EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(
                      _isLoginMode ? 'Register' : 'Sign In',
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 15,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

// Data classes for visual elements
class CreativeParticle {
  double x, y, size, speed, oscillation, phase;
  Color color;

  CreativeParticle({
    required this.x,
    required this.y,
    required this.size,
    required this.speed,
    required this.color,
    required this.oscillation,
    required this.phase,
  });
}

class FloatingElement {
  double x, y, size, speed, rotationSpeed, opacity, glowIntensity;
  IconData icon;

  FloatingElement({
    required this.x,
    required this.y,
    required this.icon,
    required this.size,
    required this.speed,
    required this.rotationSpeed,
    required this.opacity,
    required this.glowIntensity,
  });
}

// Custom painters
class CreativeParticlePainter extends CustomPainter {
  final List<CreativeParticle> particles;
  final double animationValue;

  CreativeParticlePainter(this.particles, this.animationValue);

  @override
  void paint(Canvas canvas, Size size) {
    if (size.width <= 0 || size.height <= 0) return;

    for (var particle in particles) {
      final currentY = (particle.y + animationValue * particle.speed) % 1.0;
      final currentX = particle.x +
          sin(animationValue * 2 * pi * particle.oscillation + particle.phase) *
              0.08;

      if (currentX < 0 || currentX > 1 || currentY < 0 || currentY > 1) {
        continue;
      }

      final paint = Paint()
        ..color = particle.color
        ..style = PaintingStyle.fill;

      final glowPaint = Paint()
        ..color = particle.color.withOpacity(0.2)
        ..maskFilter = MaskFilter.blur(BlurStyle.normal, particle.size * 1.5);

      final position = Offset(currentX * size.width, currentY * size.height);

      // Draw glow
      canvas.drawCircle(position, particle.size * 2, glowPaint);
      // Draw particle
      canvas.drawCircle(position, particle.size, paint);
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}

class FloatingElementPainter extends CustomPainter {
  final List<FloatingElement> elements;
  final double animationValue;

  FloatingElementPainter(this.elements, this.animationValue);

  @override
  void paint(Canvas canvas, Size size) {
    if (size.width <= 0 || size.height <= 0) return;

    for (var element in elements) {
      final currentY = (element.y + animationValue * element.speed) % 1.0;
      final currentX =
          element.x + sin(animationValue * 2 * pi * 0.3 + element.y * 8) * 0.04;

      if (currentX < 0 || currentX > 1 || currentY < 0 || currentY > 1) {
        continue;
      }

      canvas.save();
      canvas.translate(currentX * size.width, currentY * size.height);
      canvas.rotate(animationValue * 2 * pi * element.rotationSpeed);

      // Create glow effect
      final glowPaint = Paint()
        ..color = Colors.white.withOpacity(element.glowIntensity)
        ..maskFilter = MaskFilter.blur(BlurStyle.normal, element.size);

      final textPainter = TextPainter(
        text: TextSpan(
          text: String.fromCharCode(element.icon.codePoint),
          style: TextStyle(
            fontFamily: element.icon.fontFamily,
            fontSize: element.size,
            color: Colors.white.withOpacity(element.opacity),
          ),
        ),
        textDirection: TextDirection.ltr,
      );

      textPainter.layout();
      final offset = Offset(-textPainter.width / 2, -textPainter.height / 2);

      // Draw glow
      textPainter.paint(canvas, offset);

      canvas.restore();
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}
