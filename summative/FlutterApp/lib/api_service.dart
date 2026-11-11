import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

/// Base URL of the deployed FastAPI service (Render). Change it here only.
const String kApiBaseUrl = 'https://farmgate-api.onrender.com';

/// Render's free tier sleeps when idle; the first request can take ~1 minute.
const Duration kRequestTimeout = Duration(seconds: 90);

class PredictionResult {
  final double price;
  final String unit;

  const PredictionResult({required this.price, required this.unit});
}

/// An error with a message that can be shown to the user as-is.
class ApiException implements Exception {
  final String message;

  const ApiException(this.message);

  @override
  String toString() => message;
}

const Map<String, String> _fieldLabels = {
  'commodity': 'Commodity',
  'market': 'Market',
  'pricetype': 'Price type',
  'year': 'Year',
  'month': 'Month',
};

Future<PredictionResult> predictPrice({
  required String commodity,
  required String market,
  required String pricetype,
  required int year,
  required int month,
  http.Client? client,
}) async {
  final httpClient = client ?? http.Client();
  try {
    final response = await httpClient
        .post(
          Uri.parse('$kApiBaseUrl/predict'),
          headers: const {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
          body: jsonEncode({
            'commodity': commodity,
            'market': market,
            'pricetype': pricetype,
            'year': year,
            'month': month,
          }),
        )
        .timeout(kRequestTimeout);

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      return PredictionResult(
        price: (data['predicted_price_rwf'] as num).toDouble(),
        unit: data['unit'] as String,
      );
    }
    if (response.statusCode == 422) {
      throw ApiException(parseValidationError(response.body));
    }
    throw ApiException(
        'The server returned an error (${response.statusCode}). Please try again.');
  } on TimeoutException {
    throw const ApiException(
        'The server took too long to respond (it may be waking up). Please try again.');
  } on SocketException {
    throw const ApiException(
        'Could not reach the server. Check your internet connection and try again.');
  } on http.ClientException {
    throw const ApiException(
        'Could not reach the server. Check your internet connection and try again.');
  } on FormatException {
    throw const ApiException('The server sent an unexpected response. Please try again.');
  } finally {
    if (client == null) httpClient.close();
  }
}

/// Turns FastAPI's 422 body ({"detail": [{"loc": [...], "msg": ..., "type": ...}]})
/// into one friendly line per problem.
String parseValidationError(String body) {
  try {
    final detail = (jsonDecode(body) as Map<String, dynamic>)['detail'];
    if (detail is! List || detail.isEmpty) return 'Some values are invalid.';
    final lines = detail.map((item) {
      final error = item as Map<String, dynamic>;
      final loc = (error['loc'] as List?) ?? const [];
      final field = loc.isNotEmpty ? loc.last.toString() : '';
      final label = _fieldLabels[field] ?? field;
      switch (error['type']) {
        case 'missing':
          return '$label is required.';
        case 'literal_error':
          return '$label is not one of the allowed values.';
        case 'int_type':
        case 'int_parsing':
          return '$label must be a whole number.';
        case 'string_type':
          return '$label must be text.';
        default:
          return '$label: ${error['msg']}.';
      }
    }).toSet();
    return lines.join('\n');
  } catch (_) {
    return 'Some values are invalid.';
  }
}
