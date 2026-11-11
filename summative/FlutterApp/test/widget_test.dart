import 'package:farm_price_predictor/api_service.dart';
import 'package:farm_price_predictor/main.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('page has 5 input fields and a Predict button', (tester) async {
    await tester.pumpWidget(const FarmPriceApp());
    expect(find.byType(TextField), findsNWidgets(5));
    for (final label in ['Commodity', 'Market', 'Price type', 'Year', 'Month']) {
      expect(find.widgetWithText(TextField, label), findsOneWidget);
    }
    expect(find.widgetWithText(FilledButton, 'Predict'), findsOneWidget);
  });

  testWidgets('client-side validation shows readable errors', (tester) async {
    await tester.pumpWidget(const FarmPriceApp());
    await tester.enterText(find.widgetWithText(TextField, 'Commodity'), 'maiz');
    await tester.enterText(find.widgetWithText(TextField, 'Year'), 'abc');
    await tester.enterText(find.widgetWithText(TextField, 'Month'), '13');
    await tester.ensureVisible(find.text('Predict'));
    await tester.tap(find.text('Predict'));
    await tester.pump();

    final result = tester.widget<Text>(find.byKey(const Key('resultText'))).data!;
    expect(result, contains('Commodity "maiz" is not an allowed value. Did you mean: Maize'));
    expect(result, contains('Market is required.'));
    expect(result, contains('Year must be a whole number'));
    expect(result, contains('Month must be between 1 and 12.'));
  });

  test('422 response is turned into friendly text', () {
    const body = '{"detail":[{"type":"less_than_equal","loc":["body","month"],'
        '"msg":"Input should be less than or equal to 12"},'
        '{"type":"missing","loc":["body","market"],"msg":"Field required"},'
        '{"type":"literal_error","loc":["body","commodity"],"msg":"Input should be ..."}]}';
    expect(
      parseValidationError(body),
      'Month: Input should be less than or equal to 12.\n'
      'Market is required.\n'
      'Commodity is not one of the allowed values.',
    );
  });
}
