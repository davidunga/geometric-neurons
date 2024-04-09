from datetime import timedelta


class timediff:

    _full_names = {'s': 'seconds', 'm': 'minutes', 'h': 'hours', 'd': 'days'}

    @staticmethod
    def convert(src: str | timedelta, dst: str) -> float:
        """
        examples:
            convert('2d','s')
            convert('2d','2s')
            convert('2d 3h','h')
        """
        return timediff.to_timedelta(src).total_seconds() / timediff.to_timedelta(dst).total_seconds()

    @staticmethod
    def to_dict(t: str | timedelta, full_names: bool = True) -> dict[str, int]:
        t = timediff.to_timedelta(t)
        ret = {'seconds': t.seconds, 'minutes': 0, 'hours': 0, 'days': t.days}
        ret['hours'] = ret['seconds'] // 3600
        ret['seconds'] -= 3600 * ret['hours']
        ret['minutes'] = ret['seconds'] // 60
        ret['seconds'] -= 60 * ret['minutes']
        if not full_names:
            ret = {k[0]: v for k, v in ret.items()}
        return ret

    @staticmethod
    def to_timedelta(t: str | timedelta) -> timedelta:
        """
        examples:
            to_timedelta('2.5h 2d')
            to_timedelta('d')
            to_timedelta('5m 9s')
        """
        if isinstance(t, timedelta):
            return t
        kws = {}
        for tt in t.split(" "):
            value = 1.0 if len(tt) == 1 else float(tt[:-1])
            unit = tt[-1].lower()
            kws[timediff._full_names[unit]] = value
        return timedelta(**kws)


# ----------------------------

def run_test():
    TASKS = [
        dict(src='2d 3h', dst='h', gt=2 * 24 + 3),
        dict(src='m', dst='2s', gt=30),
    ]

    def _do_task(task, inv):
        src = task['src']
        dst = task['dst']
        gt = task['gt']
        if inv:
            src, dst = dst, src
            gt = 1 / gt
        res = timediff.convert(src, dst)
        is_ok = abs(gt - res) < 1e-6 * gt
        print(src, '->', dst, '=', res, "(correct)" if is_ok else "(WRONG!))")
        return is_ok

    is_oks = []
    for task in TASKS:
        is_oks.append(_do_task(task, inv=False))
        is_oks.append(_do_task(task, inv=True))

    print("TEST " + ("PASSED" if all(is_oks) else "FAILED"))


if __name__ == "__main__":
    run_test()


