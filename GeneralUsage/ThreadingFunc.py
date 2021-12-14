# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 18:24:36 2021

@author: xinmeng
"""

import threading


def run_in_thread(fn, *args, **kwargs):
    """
    Send target function to thread.
    Usage: lambda: self.run_in_thread(self.fn)

    Parameters
    ----------
    fn : function
        Target function to put in thread.

    Returns
    -------
    thread : TYPE
        Threading handle.

    """
    thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
    thread.start()
