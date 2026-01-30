class SettingNotFoundError(KeyError):
    """This setting does not exist"""


def read_values(values: dict, settingsdict: dict):
    """read values from dict into settings items

    converts dicts inside values into SettingsItem and updates settingsdict
    """
    for key, value in values.items():
        if type(value) is dict:
            settingsdict[key] = SettingsItem(value)
        else:
            settingsdict[key] = value


class SettingsItem:
    def __init__(self, defaults, settings=None):
        self._dict = {}
        if settings is not None:
            read_values(settings, self._dict)

        self.defaults = {}
        read_values(defaults, self.defaults)

    def __repr__(self):
        return f"SettingsItem({self.defaults}, {self._dict})"

    def __getitem__(self, key):
        try:
            return self._dict[key]
        except KeyError:
            pass

        try:
            return self.defaults[key]
        except KeyError:
            raise SettingNotFoundError(f"{key} not found") from None

    def __setitem__(self, key, value):
        if key in self._dict:
            self._dict[key] = value
        else:
            raise SettingNotFoundError(f"{key} not found")

    def items(self):
        for key in self.defaults:
            yield key, self[key]


class SettingsSingleton:
    def __init__(self, settings=None):
        self._dict = {}
        if settings is not None:
            read_values(settings, self._dict)

    def __repr__(self):
        return f"SettingsSingleton({self._dict})"

    def __getitem__(self, key):
        try:
            return self._dict[key]
        except KeyError:
            raise SettingNotFoundError(f"{key} not found") from None

    def items(self):
        return self._dict.items()

    def add(self, key, defaults):
        item = SettingsItem(defaults)
        self._dict[key] = item
        return item

    def save(self):
        pass  # TODO

    def restore(self):
        pass  # TODO


def settings():
    """get the settings singleton"""
    global _settings
    try:
        return _settings
    except NameError:
        _settings = SettingsSingleton()

    return _settings
