import 'package:flutter/material.dart';

import 'allowed_values.dart';
import 'api_service.dart';

void main() => runApp(const FarmPriceApp());

class FarmPriceApp extends StatelessWidget {
  const FarmPriceApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Farm Price Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: Colors.green, useMaterial3: true),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final _commodity = TextEditingController();
  final _market = TextEditingController();
  final _pricetype = TextEditingController();
  final _year = TextEditingController();
  final _month = TextEditingController();

  bool _loading = false;
  String? _result;
  bool _isError = false;

  @override
  void dispose() {
    for (final c in [_commodity, _market, _pricetype, _year, _month]) {
      c.dispose();
    }
    super.dispose();
  }

  /// Returns the canonical spelling of [input] from [allowed] (case-insensitive), or null.
  String? _match(String input, List<String> allowed) {
    final wanted = input.trim().toLowerCase();
    for (final value in allowed) {
      if (value.toLowerCase() == wanted) return value;
    }
    return null;
  }

  String _notAllowed(String label, String input, List<String> allowed) {
    final typed = input.trim().toLowerCase();
    final suggestions =
        allowed.where((v) => v.toLowerCase().contains(typed)).take(3).toList();
    final hint = suggestions.isEmpty ? '' : ' Did you mean: ${suggestions.join(', ')}?';
    return '$label "${input.trim()}" is not an allowed value.$hint';
  }

  String? _checkInt(String label, String input, int min, int max) {
    if (input.trim().isEmpty) return '$label is required.';
    final value = int.tryParse(input.trim());
    if (value == null) return '$label must be a whole number (e.g. ${label == 'Year' ? 2025 : 3}).';
    if (value < min || value > max) return '$label must be between $min and $max.';
    return null;
  }

  Future<void> _predict() async {
    FocusScope.of(context).unfocus();
    final errors = <String>[];

    String? commodity, market, pricetype;
    if (_commodity.text.trim().isEmpty) {
      errors.add('Commodity is required.');
    } else {
      commodity = _match(_commodity.text, kCommodities);
      if (commodity == null) errors.add(_notAllowed('Commodity', _commodity.text, kCommodities));
    }
    if (_market.text.trim().isEmpty) {
      errors.add('Market is required.');
    } else {
      market = _match(_market.text, kMarkets);
      if (market == null) errors.add(_notAllowed('Market', _market.text, kMarkets));
    }
    if (_pricetype.text.trim().isEmpty) {
      errors.add('Price type is required.');
    } else {
      pricetype = _match(_pricetype.text, kPriceTypes);
      if (pricetype == null) errors.add('Price type must be Retail or Wholesale.');
    }
    final yearError = _checkInt('Year', _year.text, kYearMin, kYearMax);
    final monthError = _checkInt('Month', _month.text, kMonthMin, kMonthMax);
    if (yearError != null) errors.add(yearError);
    if (monthError != null) errors.add(monthError);

    if (errors.isNotEmpty) {
      setState(() {
        _result = errors.join('\n');
        _isError = true;
      });
      return;
    }

    setState(() {
      _loading = true;
      _result = null;
    });
    try {
      final prediction = await predictPrice(
        commodity: commodity!,
        market: market!,
        pricetype: pricetype!,
        year: int.parse(_year.text.trim()),
        month: int.parse(_month.text.trim()),
      );
      setState(() {
        _result = 'Expected price: ${_formatRwf(prediction.price)} RWF/${prediction.unit}';
        _isError = false;
      });
    } on ApiException catch (e) {
      setState(() {
        _result = e.message;
        _isError = true;
      });
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  String _formatRwf(double price) {
    final digits = price.round().toString();
    return digits.replaceAllMapped(RegExp(r'\B(?=(\d{3})+(?!\d))'), (_) => ',');
  }

  Widget _field({
    required TextEditingController controller,
    required String label,
    required String helper,
    bool numeric = false,
    TextInputAction action = TextInputAction.next,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: TextField(
        controller: controller,
        keyboardType: numeric ? TextInputType.number : TextInputType.text,
        textCapitalization: numeric ? TextCapitalization.none : TextCapitalization.words,
        textInputAction: action,
        decoration: InputDecoration(
          labelText: label,
          helperText: helper,
          helperMaxLines: 3,
          border: const OutlineInputBorder(),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Farm Price Predictor'),
        backgroundColor: scheme.primaryContainer,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                'Estimate the market price of a commodity in a Rwandan market for a given month.',
                style: Theme.of(context).textTheme.bodyMedium,
              ),
              const SizedBox(height: 20),
              _field(
                controller: _commodity,
                label: 'Commodity',
                helper: '${kCommodities.length} allowed, e.g. Maize, Beans (dry), '
                    'Potatoes (Irish), Cassava, Rice, Bananas',
              ),
              _field(
                controller: _market,
                label: 'Market',
                helper: '${kMarkets.length} allowed, e.g. Kimironko, Nyabugogo, '
                    'Musanze, Huye, Rubavu, Byumba',
              ),
              _field(
                controller: _pricetype,
                label: 'Price type',
                helper: 'Retail or Wholesale (wholesale data is limited)',
              ),
              _field(
                controller: _year,
                label: 'Year',
                helper: 'Whole number from $kYearMin to $kYearMax',
                numeric: true,
              ),
              _field(
                controller: _month,
                label: 'Month',
                helper: 'Whole number from 1 (January) to 12 (December)',
                numeric: true,
                action: TextInputAction.done,
              ),
              const SizedBox(height: 4),
              SizedBox(
                height: 48,
                child: FilledButton(
                  onPressed: _loading ? null : _predict,
                  child: const Text('Predict'),
                ),
              ),
              const SizedBox(height: 20),
              _ResultArea(loading: _loading, text: _result, isError: _isError),
            ],
          ),
        ),
      ),
    );
  }
}

/// The single area that shows the loading state, the predicted price, or an error.
class _ResultArea extends StatelessWidget {
  final bool loading;
  final String? text;
  final bool isError;

  const _ResultArea({required this.loading, required this.text, required this.isError});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final textTheme = Theme.of(context).textTheme;

    Widget child;
    if (loading) {
      child = const Row(
        children: [
          SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 3)),
          SizedBox(width: 16),
          Expanded(
            child: Text('Getting prediction... The first request can take up to a minute '
                'while the server wakes up.'),
          ),
        ],
      );
    } else if (text == null) {
      child = Text('Fill in all fields and tap Predict.',
          style: textTheme.bodyMedium?.copyWith(color: scheme.onSurfaceVariant));
    } else {
      child = Text(
        text!,
        key: const Key('resultText'),
        style: isError
            ? textTheme.bodyMedium?.copyWith(color: scheme.error)
            : textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w600),
      );
    }

    return Container(
      width: double.infinity,
      constraints: const BoxConstraints(minHeight: 72),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: isError && !loading && text != null
            ? scheme.errorContainer.withValues(alpha: 0.4)
            : scheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
      ),
      child: child,
    );
  }
}
