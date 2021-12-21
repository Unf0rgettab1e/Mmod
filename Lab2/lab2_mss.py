import os
import logging
import random
import math

import numpy as np
import pandas as pd
import markdown_generator as mg
import matplotlib.pyplot as plt

from datetime import datetime


def get_free_probability(p, n, m, beta):
    p0 = 1

    for k in range(1, n + 1):
        p0 += p ** k / math.factorial(k)

    sum = 0
    i = 1
    while i <= m:
        p_up = p ** i
        l = 1
        compos = 1
        while l <= i:
            compos *= (n + l * beta)
            l += 1
        sum += p_up / compos
        i += 1

    p0 += ((p ** n) / math.factorial(n)) * sum
    return 1 / p0


def get_pn_probability(p, n, m, beta):
    p0 = get_free_probability(p, n, m, beta)
    return p0 * (p ** n) / math.factorial(n)


def get_state_probs(rho, n, m, beta):
    p0 = get_free_probability(rho, n, m, beta)
    probs = [p0]

    for k in range(1, n + 1):
        probs.append(rho ** k / math.factorial(k) * p0)

    pn = get_pn_probability(rho, n, m, beta)
    for i in range(1, m + 1):
        l = 1
        compos = 1
        while l <= i:
            compos *= (n + l * beta)
            l += 1
        probs.append(pn * (rho ** i) / compos)

    return probs


def get_cancel_prob(p, n, m, beta):
    pn = get_pn_probability(p, n, m, beta)
    l = 1
    denominator = 1
    while l <= m:
        denominator *= (n + l * beta)
        l += 1
    return pn * (p ** m) / denominator


def get_theor_interval_len(n, m, probs):
    sum = 0
    for i in range(0, m):
        sum += ((i + 1) * probs[n + 1 + i])
    return sum


def get_theor_channel_loaded(n, m, probs):
    sum = 0

    for k in range(0, n):
        sum += probs[k + 1] * (k + 1)

    for i in range(0, m):
        sum += n * probs[n + i + 1]

    return sum


def get_overall_request_time(theor_relative_bandwidth, mu, t):
    return theor_relative_bandwidth / mu + t




class Channel:
    COUNT = 0

    def __init__(self, mu, reject_probability, on_reject):
        Channel.COUNT += 1
        self.id = Channel.COUNT

        self.reject_probability = reject_probability
        self.on_reject = on_reject
        self.mu = mu

        self.free = True
        self.end_at = 0
        self.request = None

    def run(self, start_at, request):
        self.free = False
        self.end_at = start_at + np.random.exponential(1 / self.mu)
        self.request = request
        logging.info('[Стартовал] Канал #%d: с %.4f до %.4f' %
                     (self.id, start_at, self.end_at))

    def try_free(self, cur_time):
        if not self.free and self.end_at < cur_time:
            self.free = True

            rejected = random.random() <= self.reject_probability
            if rejected:
                self.on_reject(self.request)
                logging.info('[Отклонено] Канал #%d: освободился в %.4f' %
                             (self.id, cur_time))
            else:
                logging.info('[Выполнено] Канал #%d: осободился в %.4f' %
                             (self.id, cur_time))
                return self.request


##---------------------------------------------------------------------------------------


class Request:
    COUNT = 0

    def __init__(self, cur_time):
        Request.COUNT += 1
        self.id = Request.COUNT
        self.time_in_queue = 0
        self.time_in_system = 0
        self.start_in_queue = None
        self.start_in_system = cur_time
        self.time_waiting_in_queue = None

    def enqueue(self, cur_time, v):
        self.start_in_queue = cur_time
        self.time_waiting_in_queue = self.start_in_queue + np.random.exponential(1 / v)
        logging.info('[Поступил] Запрос #%d в %.4f' % (self.id, cur_time))

    def dequeue(self, cur_time):
        self.time_in_queue += cur_time - self.start_in_queue
        logging.info('[Выполняется] Запрос #%d в %.4f' % (self.id, cur_time))

    def out(self, cur_time):
        self.time_in_system += cur_time - self.start_in_system

    def __str__(self):
        return 'Запрос #%d: в очереди %.4f, в системе %.4f' % \
               (self.id, self.time_in_queue, self.time_in_system)


##---------------------------------------------------------------------------------------


class Stats:
    def __init__(self, system):
        self.system = system

        self.queue_sizes = []
        self.working_channels = []
        self.total_requests = []
        self.requests = []
        self.request_queue_times = []
        self.request_times = []

        self.work_intervals = []
        self.process_intervals = []

        self.times_graphics = []
        self.finished_req_graphics = []
        self.cancelled_req_graphics = []
        self.running_req_graphics = []
        self.queue_req_graphics = []

        self.rejections = 0
        self.cancellations = 0

    def collect(self):
        cur_working_channels = 0
        for channel in self.system.channels:
            cur_working_channels += not channel.free

        cur_queue_size = len(self.system.queue)

        self.queue_sizes.append(cur_queue_size)
        self.working_channels.append(cur_working_channels)
        self.total_requests.append(cur_queue_size + cur_working_channels)

    def collect_for_graphics(self, cur_time, running_req, queue_size):
        self.times_graphics.append(cur_time)
        self.finished_req_graphics.append(len(self.requests))
        self.cancelled_req_graphics.append(self.cancellations)
        self.running_req_graphics.append(running_req)
        self.queue_req_graphics.append(queue_size)

    def cancel(self):
        self.cancellations += 1

    def reject(self):
        self.rejections += 1

    def add_work(self, interval):
        self.work_intervals.append(interval)

    def add_process(self):
        self.process_intervals.append(self)

    def out(self, request):
        self.requests.append(request)
        self.request_queue_times.append(request.time_in_queue)
        self.request_times.append(request.time_in_system)

    def build(self):
        d = {'Размер очереди': self.queue_sizes,
             'Занятые каналы': self.working_channels,
             'Заявки в системе': self.total_requests}

        d1 = {'Время запроса в очереди': self.request_queue_times,
              'Время запроса в системе': self.request_times}

        return pd.DataFrame(data=d), pd.DataFrame(data=d1)

    def get_cancel_prob(self):
        return self.cancellations / self.system.request_limit

    def get_states_probs(self):
        states = list(i for i in range(self.system.n + 1))
        states += list(i for i in range(self.system.n + 1, self.system.n + self.system.m + 1))

        state_counts = np.zeros(len(states))

        for req in self.total_requests:
            state_counts[req] += 1

        return states, state_counts

##---------------------------------------------------------------------------------------

class System:
    def __init__(self, n, m, lambda_, mu, p, tick_size, request_limit, t):
        self.n = n
        self.m = m
        self.lambda_ = lambda_
        self.mu = mu
        self.p = p
        self.q = 1 - p
        self.tick_size = tick_size
        self.request_limit = request_limit
        self.t = t

        self.stats = Stats(self)

        self.request = 0
        self.channels = [
            Channel(self.mu, self.q, self.request_rejected) for _ in range(n)
        ]
        self.queue = []
        self.cur_time = 0.
        self.next_time = np.random.exponential(lambda_)

    def log(self):
        logging.info('Текущее время %.4f, следующий запрос поступит %.4f' %
                     (self.cur_time, self.next_time))

    def request_rejected(self, request=None):
        self.stats.reject()
        if request:
            self.push(request)

    def push(self, request=None):
        if not request:
            self.request += 1
            request = Request(self.cur_time)

        if len(self.queue) < self.m:
            request.enqueue(self.cur_time, 1. / self.t)
            self.queue.append(request)
        else:
            self.stats.cancel()
            logging.info('[Отменён] %s' % (request,))

    def free_channels(self):
        for channel in self.channels:
            request = channel.try_free(self.cur_time)
            if request:
                request.out(self.cur_time)
                self.stats.out(request)

    def dequeue_requests(self):
        for channel in self.channels:
            if not len(self.queue):
                return

            if channel.free:
                request = self.queue.pop(0)
                request.dequeue(self.cur_time)
                channel.run(self.cur_time, request)

    def free_all(self):
        while not self.is_all_free():
            self.tick()

    def is_all_free(self):
        for channel in self.channels:
            if not channel.free:
                return False
        return True

    def tick(self):
        self.free_channels()
        self.dequeue_requests()
        self.stats.collect()

        channels_in_work = 0
        for channel in self.channels:
            if not channel.free:
                channels_in_work += 1
        self.stats.collect_for_graphics(self.cur_time, channels_in_work, len(self.queue))

        self.cur_time += self.tick_size

    def check_queue(self):
        if not len(self.queue):
            return
        for request in self.queue:
            if self.cur_time >= request.time_waiting_in_queue:
                self.queue.remove(request)

    def run(self):
        while self.request < self.request_limit:
            if self.cur_time >= self.next_time:
                self.next_time = self.cur_time + \
                                 np.random.exponential(1 / self.lambda_)
                self.push()
                self.log()

            self.check_queue()
            self.tick()

        self.free_all()


##---------------------------------------------------------------------------------------


logging.basicConfig(
    format='%(message)s',
    level=logging.INFO
)


def generate_report(lambda_, mu, p, n, m, t, stats):
    path = 'results'
    hists_dir_name = 'hists'
    hists_path = os.path.join(path, hists_dir_name)
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(hists_path):
        os.makedirs(hists_path)
    time_now = datetime.now().strftime('%d%m%Y_%H%M%S')
    filename = 'result_%s.md' % (time_now,)
    hist_name = time_now + '.png'
    hist_name_2 = time_now + '-2' + '.png'
    hist_name_3 = time_now + '-3' + '.png'

    with open(os.path.join(path, filename), 'w', encoding='utf-8') as f:
        doc = mg.Writer(f)
        doc.write_heading('Статистика')
        doc.writelines([
            'λ = %.2f' % lambda_, '',
            'μ = %.2f' % mu, '',
            'n = %d, m = %d' % (n, m), ''
        ])

        df_c, df_times = stats.build()
        doc.writelines([
            df_c.describe().T.to_markdown(), '',
        ])

        doc.writelines([
            'Всего отменено: %d' % stats.cancellations, '',
            'Всего выполнено: %d' % len(stats.requests), ''
        ])

        states_bins, states_counts = stats.get_states_probs()
        _rho = lambda_ / mu

        plt.xticks(states_bins)
        plt.hist(stats.total_requests, bins=np.array(states_bins) - 0.5, density=True)
        plt.savefig(os.path.join(hists_path, hist_name))

        beta = 1 / (t * mu)
        probs = get_state_probs(_rho, n, m, beta)

        doc.writelines([
            'Вероятности для состояний системы:',
            '![hist](%s)' % (os.path.join(hists_dir_name, hist_name),), '',
            pd.DataFrame(data={
                'Теоретическая вероятность': probs,
                'Практическая вероятность': states_counts / sum(states_counts)
            }).T.to_markdown(), ''
        ])

        plt.plot(stats.times_graphics, stats.finished_req_graphics, label="Finished")
        plt.plot(stats.times_graphics, stats.cancelled_req_graphics, label="Cancelled")
        plt.legend()
        plt.savefig(os.path.join(hists_path, hist_name_2))

        doc.writelines([
            'Данный график демонстрирует рост числа выполненных и отменённых заявок со временем:',
            '![graph](%s)' % (os.path.join(hists_dir_name, hist_name_2),), ''
        ])

        plt.clf()
        plt.plot(stats.times_graphics, stats.running_req_graphics, label="Running")
        plt.plot(stats.times_graphics, stats.queue_req_graphics, label="In queue")
        plt.xlim(max(stats.times_graphics) - 20, max(stats.times_graphics))
        plt.legend()
        plt.savefig(os.path.join(hists_path, hist_name_3))

        doc.writelines([
            'Данный график демонстрирует количество заявок в каналах и очереди в течение времени выполнения:',
            '![graph](%s)' % (os.path.join(hists_dir_name, hist_name_3),), ''
        ])

        cancel_prob = stats.get_cancel_prob()
        theor_cancel_prob = get_cancel_prob(_rho, n, m, beta)
        relative_bandwidth = 1 - cancel_prob
        theor_relative_bandwidth = 1 - theor_cancel_prob
        absolute_bandwidth = relative_bandwidth * lambda_
        theor_absolute_bandwidth = lambda_ * theor_relative_bandwidth
        theor_queue_size = get_theor_interval_len(n, m, probs)
        theor_channel_loaded = get_theor_channel_loaded(n, m, probs)
        theor_system_load = theor_queue_size + theor_channel_loaded

        doc.writelines([
            pd.DataFrame({
                'Вероятность отказа': [theor_cancel_prob, cancel_prob],
                'Относительная пропускная способность': [theor_relative_bandwidth, relative_bandwidth],
                'Абсолютная пропускная способность': [theor_absolute_bandwidth, absolute_bandwidth],
                'Длина очереди': [theor_queue_size, np.mean(stats.queue_sizes)],
                'Количество занятых каналов': [theor_channel_loaded, np.mean(stats.working_channels)],
                'Количество заявок в системе': [theor_system_load, np.mean(stats.total_requests)],
            }, index=['Теор.', 'Практ.']).T.to_markdown(), ''
        ])

        doc.writelines([
            df_times.describe().T.to_markdown(), ''
        ])

        theor_overall_request_time = get_overall_request_time(theor_relative_bandwidth, mu, t)

        doc.writelines([
            pd.DataFrame({
                'Теор. среднее время пребывания заявки в СМО': [theor_overall_request_time]
            }, index=['Значение']).T.to_markdown(), ''
        ])

        plt.clf()


lambda_ = [16, 24]
mu = [6, 4.8]
p = 1
n = [5, 9]
m = [6, 8]
t = 1. / 8.
for i in range(0, len(lambda_)):
    system = System(n[i], m[i], lambda_[i], mu[i], p, 0.01, 10000, t)
    system.log()
    system.run()
    generate_report(lambda_[i], mu[i], p, n[i], m[i], t, system.stats)
