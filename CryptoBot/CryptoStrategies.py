import numpy as np

class Strategy:

    def prediction_stat(self, predictions):
        if len(predictions) > 0:
            prediction = np.mean(predictions)
            order_std = np.std(predictions)
        else:
            prediction = 0
            order_std = 100

        return prediction, order_std

    def condition_prediction(self, side, predictions, prices):

        if side == 'bids':
            coeff = -1
        else:
            coeff = 1


        price_mask = np.abs( prices - prices[-1] ) < 3 * np.std(prices) * np.ones(prices.shape)

        if len(price_mask) == len(predictions):
            norm_predictions = coeff * predictions[price_mask]
        else:
            norm_predictions = coeff * predictions[price_mask[1::]]
        plus_predictions = norm_predictions[norm_predictions > 0.01]
        minus_predictions = norm_predictions[norm_predictions < -0.01]

        plus_prediction, plus_std = self.prediction_stat(plus_predictions)
        minus_prediction, minus_std = self.prediction_stat(minus_predictions)

        return plus_prediction, minus_prediction, coeff, plus_std, minus_std


    def determine_move(self, predictions, order_book, portfolio, bids, asks):

        is_holding_usd = portfolio.value['USD'] > 10

        if is_holding_usd:
            side = 'bids'
            prices = bids
            opposing_prices = asks
        else:
            side = 'asks'
            prices = asks
            opposing_prices = bids

        current_price = prices[-1]
        current_opposing_price = opposing_prices[-1]
        prediction, opposing_prediction, coeff, plus_std, minus_std = self.condition_prediction(side, predictions, prices)
        plus_del = prediction
        minus_del = opposing_prediction
        plus_price = current_price + coeff * ( prediction)
        minus_price = current_opposing_price - coeff * 0.01

        if (plus_std < minus_std):#(plus_del > (0.0015 * plus_price)):
            decision = {'side': side, 'size coeff': 1, 'price': plus_price, 'is maker': True}
        elif (plus_std > minus_std):#(minus_del < (-0.0025 * minus_price)):
            decision = {'side': side, 'size coeff': 1, 'price': minus_price, 'is maker': False}
        else:
            decision = None

        # decision = {'side': side, 'size coeff': 1, 'price': plus_price, 'is maker': True}
        # decision = {'side': 'bids', 'size coeff': 1, 'price': 128}

        return decision, plus_std