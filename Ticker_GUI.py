import numpy as np
import pandas as pd
import dearpygui.dearpygui as dpg
import robin_stocks.robinhood as r
from dotenv import load_dotenv
import os
import pickle
import datetime
# import time

# not done yet
class IndicatorSet:
    def __init__(self, indicator : tuple, interval : int, span : int, specific_symbol=None):
        self.indicator = indicator
        self.interval = interval
        self.span = span
        self.specific_symbol = specific_symbol


def _log(sender, app_data, user_data):
    print(f"sender: {sender}, \t app_data: {app_data}, \t user_data: {user_data}")

# works. takes a list named saved_symbols and stores the data to a pickle in folder for later use
def add_or_delete_to_saved_symbol(new_symbol=None, add=None):
    global saved_symbols
    if new_symbol is not None and add == True:
        saved_symbols.append(new_symbol)
    elif new_symbol is not None and add == False:
        for n, sym in enumerate(saved_symbols):
            if new_symbol == sym:
                saved_symbols.pop(n)
                break

    saved_symbols = list(set(saved_symbols))
    # sort before returning. To look mor propheshional
    saved_symbols.sort()
    with open('symbols.pickle', 'wb') as file:
        pickle.dump(saved_symbols, file)
    dpg.configure_item("selected_symbol", items=saved_symbols)
    dpg.configure_item("win_change_syms", show=False)
    dpg.configure_item("new_symbol", default_value="")


# works. takes pickle in folder and brings the data to a list named saved_symbols
def retrieve_symbols():
    with open('symbols.pickle', 'rb') as file:
        symbols = pickle.load(file)

    return symbols


saved_symbols = retrieve_symbols()
interval_options = ["5minute", "10minute", "hour", "day", "week"]
span_options = ["day", "week", "month", "3month", "year", "5year"]
bounds_options = ["extended", "trading", "regular"]

indicator_options = ["Moving Average", "MACD", "RSI"]

amount_of_times_window_opened = 0

set_of_indicators = []
"""with open('indicators.pickle', 'rb') as file:
    set_of_indicators = pickle.load(file)"""

# Not tested yet. saving an indicator.
def create_new_indicator(indicator, interval, span, specific_symbol=None, symbol_if_needed=None):
    global set_of_indicators
    if specific_symbol:
        specific_symbol = symbol_if_needed
    else:
        specific_symbol = None
    set_of_indicators.append(IndicatorSet(indicator, interval, span, specific_symbol))
    with open('indicators.pickle', 'wb') as file:
        pickle.dump(set_of_indicators, file)


# Robinhood login. uses dotenv to store sensitive data on an .env file. it's local but should not be on github.
# the auth code is provided using an authenticator app that refreshes every 30 seconds to a minute
def login_rh(auth=None):
    login_valid = False
    tries = 3
    print(auth)
    while not login_valid:
        try:
            load_dotenv()
            if auth is None:
                two_factor_authenticator = input("What is the current authenticator?")
            else:
                two_factor_authenticator = auth

            login_valid = r.login(os.environ['robin_username'],
                                  os.environ['robin_password'],
                                  mfa_code=two_factor_authenticator,
                                  store_session=False)
        except KeyError as e:
            print("that seems to be an incorrect authenticator. Try again...")
            tries -= 1
            if tries <= 0:
                print("you have 0 tries left to login. Please check login code.")
                print(e)
                break
    if login_valid:
        if auth is not None:
            dpg.configure_item("win_login_rh", show=False)
            dpg.configure_item("login_status", default_value="OK")
            dpg.configure_item("login_status", color=(0,255,0))
        print("login successful")

    return login_valid


# not sure if this function is needed. Has been very buggy. needs revamping
def display_popup_per_indicator(indicator, parent):
    win_tag = f"win_add_{indicator[:3]}"
    print(win_tag)
    with dpg.popup(parent, modal=True, mousebutton=dpg.mvMouseButton_Left, tag=win_tag):
        dpg.add_text(indicator)
        dpg.add_text(f"Making this indicator to only work with the interval " +
                     f"of {dpg.get_value("selected_interval")}, and " +
                     f"a span of {dpg.get_value("selected_span")}")
        if indicator == indicator_options[0]:       # moving average
            dpg.add_input_text(label="Period", hint="20", callback=_log, tag=f"moving_avg_period", decimal=True)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Exponential", callback=_log, tag=f"Exponential_MA")
                dpg.add_button(label="Create",
                               callback=lambda: create_new_indicator((indicator,
                                                                      dpg.get_value("moving_avg_period"),
                                                                      dpg.get_value("Exponential_MA")),
                                                                     dpg.get_value("selected_interval"),
                                                                     dpg.get_value("selected_span"),
                                                                     dpg.get_value(f"indicator_specific{indicator[:3]}"),
                                                                     dpg.get_value(f"specific_symbol{indicator[:3]}")))
        elif indicator == indicator_options[1]:     # MACD
            with dpg.group(horizontal=True):
                dpg.add_input_text(label="EMA1 Period", hint="12", callback=_log,
                                   tag=f"ema1_period", width=70, decimal=True)
                dpg.add_input_text(label="EMA2 Period", hint="26", callback=_log,
                                   tag=f"ema2_period", width=70, decimal=True)
                dpg.add_input_text(label="Signal Period", hint="9", callback=_log,
                                   tag=f"signal_period", width=70, decimal=True)
            dpg.add_button(label="Create", callback=lambda: create_new_indicator((indicator,
                                                                                  dpg.get_value("ema1_period"),
                                                                                  dpg.get_value("ema2_period"),
                                                                                  dpg.get_value("signal_period")),
                                                                                 dpg.get_value("selected_interval"),
                                                                                 dpg.get_value("selected_span"),
                                                                                 dpg.get_value(f"indicator_specific{indicator[:3]}"),
                                                                                 dpg.get_value(f"specific_symbol{indicator[:3]}")))
        elif indicator == indicator_options[2]:     # RSI
            dpg.add_input_text(label="Period", hint="14", callback=_log, tag=f"rsi_period", decimal=True)
            dpg.add_button(label="Create", callback=lambda: create_new_indicator((indicator,
                                                                                  dpg.get_value("rsi_period")),
                                                                                 dpg.get_value("selected_interval"),
                                                                                 dpg.get_value("selected_span"),
                                                                                 dpg.get_value(f"indicator_specific{indicator[:3]}"),
                                                                                 dpg.get_value(f"specific_symbol{indicator[:3]}")))
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="For a specific symbol", tag=f"indicator_specific{indicator[:3]}",
                             callback=lambda: dpg.configure_item(f"specific_symbol{indicator[:3]}", show=True))
            dpg.add_combo(saved_symbols, label="Which symbol?", callback=_log, tag=f"specific_symbol{indicator[:3]}",
                          fit_width=True, show=False)
    # dpg.configure_item(win_tag,show=True)

# use robinhood to get historical data to display on graph
def get_historical_rh(symbol, interval='day', span='year', bounds="regular"):
    global amount_of_times_window_opened
    hist = r.get_stock_historicals(symbol, interval, span, bounds)
    hist = pd.DataFrame(hist)
    print(hist)
    print(type(hist))
    stock_name = r.get_name_by_symbol(symbol)
    timestamp = []
    for begin in hist["begins_at"].tolist():
        timestamp.append(datetime.datetime.fromisoformat(begin).timestamp())

    window_instance = amount_of_times_window_opened
    hist_window_tag = f"{symbol}{window_instance}"
    with dpg.window(label=stock_name, width=1000, height=400, pos=(300, 0),
                    tag= hist_window_tag, on_close=lambda: dpg.delete_item(hist_window_tag)):
        amount_of_times_window_opened += 1

        # need to add the ability to also view pre-selected indicators as well
        with dpg.plot(label="Candle Series", height=400, width=-1):
            dpg.add_plot_legend()
            xaxis = dpg.add_plot_axis(dpg.mvXAxis, label=interval, scale=dpg.mvPlotScale_Time)
            with dpg.plot_axis(dpg.mvYAxis, label="USD"):
                dpg.add_candle_series(timestamp, hist["open_price"].astype(float).tolist(),
                                      hist["close_price"].astype(float).tolist(), hist["low_price"].astype(float).tolist(),
                                      hist["high_price"].astype(float).tolist(), label=symbol, time_unit=dpg.mvTimeUnit_Day)
                dpg.fit_axis_data(dpg.top_container_stack())
            dpg.fit_axis_data(xaxis)


# start of functions for indicators
# made these functions a while ago before chatGpt and while I was very new to programming.
# Except the RSI. I think you have to pass it a pandas df to it if I remember right
def make_moving_ave(close_list, period=20, simple=True):
    sma = []
    start = period - 1  # was 0
    # close_ = []
    while True:
        if start >= len(close_list):
            break
        average = 0.0
        for i in range(period):
            num = (start - i)
            average += close_list[num]
        average /= period
        sma.append(average)
        # close_.append(close_list[start])                # for making sure that the close price aligns with the sma
        # was if (start + period) == len(close_list): break
        start += 1

    if simple:
        pre = [np.nan] * (period - 1)
        sma = np.concatenate([pre, sma])
        # temp1 = np.concatenate([pre, close_])         # for making sure that the close price aligns with the sma
        # temp = np.array([sma, temp1, close_list])     # for making sure that the close price aligns with the sma
        return sma
    else:
        ema = []
        alpha = 0.3
        start = period - 1
        for i in sma:
            a = alpha * close_list[start]
            b = (1 - alpha) * i
            c = a + b
            ema.append(c)
            if (start + 1) == len(close_list): break
            start += 1
        if not simple:
            pre = [np.nan] * period
            ema = np.concatenate([pre, ema])
            return ema


def make_macd(close_list, ema1=12, ema2=26, ema_sig=9):
    ema12 = make_moving_ave(close_list, ema1, simple=False)
    ema26 = make_moving_ave(close_list, ema2, simple=False)

    start = len(ema12) - len(ema26)
    macd = []
    for i in range(len(ema26)):
        mac = ema12[i + start] - ema26[i]
        macd.append(mac)

    signal = make_moving_ave(macd, ema_sig, simple=False)
    start = len(macd) - len(signal)
    macd_histo = []
    for i in range(len(signal)):
        his = macd[i + start] - signal[i]
        macd_histo.append(his)

    return macd, signal, macd_histo


def calculate_rsi(close_prices, period=14):
    price_diff = close_prices.diff()
    gain = price_diff.where(price_diff > 0, 0)
    loss = -price_diff.where(price_diff < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi
# end of functions for indicators

# works pretty well at this moment. I like the way to login
def show_ticker_gui():
    FONT_SCALE = 2
    with dpg.font_registry():
        font_regular = dpg.add_font('Inter-Regular.ttf', 16 * FONT_SCALE)
        font_medium = dpg.add_font('Inter-Medium.ttf', 16 * FONT_SCALE)
        font_bold = dpg.add_font('Inter-Bold.ttf', 22 * FONT_SCALE)
    dpg.set_global_font_scale(1 / FONT_SCALE)
    dpg.bind_font(font_medium)

    with dpg.window(label='Robinhood Ticker GUI', width=300, height=300):
        with dpg.group(horizontal=True):
            dpg.add_text("X", color=(255,0,0), tag="login_status")
            dpg.bind_item_font(dpg.last_item(), font_bold)
            dpg.add_button(label="Login to Robinhood")
            dpg.bind_item_font(dpg.last_item(), font_bold)
            with dpg.popup(dpg.last_item(), modal=True, mousebutton=dpg.mvMouseButton_Left, tag="win_login_rh"):
                dpg.add_text("Use an authenticator app to retrieve a Two Factor Authenticator code")
                dpg.add_input_text(label="Authenticator", hint="enter text here", callback=_log, tag="MFA_input",
                                   decimal=True, no_spaces=True)
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_button(label="OK", width=75, callback=lambda : login_rh(dpg.get_value("MFA_input")))
                    # ^value in get_value must be the sender of the Authenticator input text^
                    dpg.add_button(label="Cancel", width=75,
                                   callback=lambda: dpg.configure_item("win_login_rh", show=False))
        dpg.add_spacer(height=15)

        with dpg.group():
            dpg.add_combo(saved_symbols, label="Symbol", default_value=saved_symbols[0],
                          callback=_log, tag="selected_symbol", fit_width=True)
            dpg.add_combo(interval_options, label="Interval", default_value=interval_options[3], callback=_log,
                          tag="selected_interval", fit_width=True)
            dpg.add_combo(span_options, label="Span", default_value=span_options[4], callback=_log,
                          tag="selected_span", fit_width=True)
            dpg.add_combo(bounds_options, label="Bounds", default_value=bounds_options[2], callback=_log,
                          tag="selected_bounds", fit_width=True)
            dpg.add_button(label="View", callback=lambda: get_historical_rh(symbol=dpg.get_value("selected_symbol"),
                                                                            interval=dpg.get_value("selected_interval"),
                                                                            span=dpg.get_value("selected_span"),
                                                                            bounds=dpg.get_value("selected_bounds")))
            dpg.add_spacer(width=40)
            # Add or delete symbols from symbol list
            dpg.add_button(label="Add/Delete Symbol", callback=_log)
            with dpg.popup(dpg.last_item(), modal=True, mousebutton=dpg.mvMouseButton_Left, tag="win_change_syms"):
                dpg.add_input_text(label="Symbol Name", hint="ABC", callback=_log, tag="new_symbol", uppercase=True)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Add",
                                   callback=lambda: add_or_delete_to_saved_symbol(dpg.get_value("new_symbol"), True))
                    dpg.add_button(label="Delete",
                                   callback=lambda: add_or_delete_to_saved_symbol(dpg.get_value("new_symbol"), False))

            # Indicator management
            dpg.add_spacer(width=40)
            dpg.add_button(label="Indicator Management", callback=_log)
            with dpg.popup(dpg.last_item(), modal=True, mousebutton=dpg.mvMouseButton_Left, tag="win_manage_ind"):
                with dpg.group(horizontal=True):
                    add_but_tag0 = f"add_{indicator_options[0][:3]}_but"
                    dpg.add_button(label=f"Add {indicator_options[0]}", tag=add_but_tag0,
                                   callback=lambda: display_popup_per_indicator(indicator_options[0], add_but_tag0))
                    add_but_tag1 = f"add_{indicator_options[1][:3]}_but"
                    dpg.add_button(label=f"Add {indicator_options[1]}", tag=add_but_tag1,
                                   callback=lambda: display_popup_per_indicator(indicator_options[1], add_but_tag1))
                    add_but_tag2 = f"add_{indicator_options[2][:3]}_but"
                    dpg.add_button(label=f"Add {indicator_options[2]}", tag=add_but_tag2,
                                   callback=lambda: display_popup_per_indicator(indicator_options[2], add_but_tag2))



dpg.create_context()
dpg.set_global_font_scale(1.25)
show_ticker_gui()
dpg.create_viewport(title='Ticker GUI', width=1200, height=1000)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
