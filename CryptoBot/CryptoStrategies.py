import numpy as np

class Strategy:

    def prediction_stat(self, predictions, include_std):
        if len(predictions) > 0:
            prediction = np.mean(predictions)
            order_std = np.std(predictions)
        else:
            prediction = 0
            order_std = 100

        return prediction, order_std

    def condition_prediction(self, side, predictions):

        if side == 'bids':
            coeff = -1
        else:
            coeff = 1

        norm_predictions = coeff * predictions
        plus_predictions = norm_predictions[norm_predictions > 0.01]
        minus_predictions = norm_predictions[norm_predictions < -0.01]

        plus_prediction, plus_std = self.prediction_stat(plus_predictions, 1)
        minus_prediction, minus_std = self.prediction_stat(minus_predictions, 0)

        return plus_prediction, minus_prediction, coeff, plus_std, minus_std


    def determine_move(self, predictions, order_book, portfolio):

        is_holding_usd = portfolio.value['USD'] > 10

        if is_holding_usd:
            side = 'bids'
            opposing_side = 'asks'
            prices = order_book['0'].values
            opposing_prices = order_book['60'].values
        else:
            side = 'asks'
            opposing_side = 'bids'
            prices = order_book['60'].values
            opposing_prices = order_book['0'].values

        current_price = prices[-1]
        current_opposing_price = opposing_prices[-1]
        prediction, opposing_prediction, coeff, plus_std, minus_std = self.condition_prediction(side, predictions)
        plus_del = prediction
        minus_del = opposing_prediction - minus_std
        plus_price = current_price + coeff * ( prediction)
        minus_price = current_opposing_price - coeff * 0.01

        if (plus_del > (0.0015 * plus_price)):
            decision = {'side': side, 'size coeff': 1, 'price': plus_price, 'is maker': True}
        elif (minus_del < (-0.0025 * minus_price)):
            decision = {'side': side, 'size coeff': 1, 'price': minus_price, 'is maker': False}
        else:
            decision = None

        # decision = {'side': side, 'size coeff': 1, 'price': plus_price, 'is maker': True}
        #decision = {'side': 'bids', 'size coeff': 1, 'price': 128}

        return decision, plus_std