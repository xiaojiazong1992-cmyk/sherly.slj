# -*- coding: utf-8 -*-
"""
data_scraper.py
数据爬取模块：从国家统计局、全国公共资源交易平台等获取真实数据
"""

import time
import json
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────
# 1. 国家统计局 NBS 分省房屋竣工造价
# ─────────────────────────────────────────────

def fetch_nbs_construction_cost(retries=3, delay=2):
    """
    从国家统计局API获取各省房屋竣工造价数据（元/㎡）
    数据集：分省年度数据，覆盖住宅类竣工造价
    指标代码：A060C01（房屋竣工造价）
    返回 DataFrame：columns=[province, year, unit_cost]
    """
    url = "https://data.stats.gov.cn/easyquery.htm"
    params = {
        'm': 'QueryData',
        'dbcode': 'fsnd',          # 分省年度数据库
        'rowcode': 'reg',          # 行：地区
        'colcode': 'sj',           # 列：时间
        'wds': '[]',
        # 指标=房屋竣工造价，时间=最近6年
        'dfwds': '[{"wdcode":"zb","valuecode":"A060C01"},'
                 '{"wdcode":"sj","valuecode":"LAST6"}]',
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://data.stats.gov.cn/easyquery.htm',
    }

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            raw = resp.json()
            return _parse_nbs_json(raw)
        except Exception as e:
            print(f"[NBS] 第{attempt+1}次尝试失败: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))  # 指数退避
    print("[NBS] 警告：所有重试均失败，返回空DataFrame")
    return pd.DataFrame(columns=['province', 'year', 'unit_cost'])


def _parse_nbs_json(raw):
    """解析NBS返回的JSON数据，提取省份、年份、造价"""
    records = []
    try:
        nodes = raw['returndata']['datanodes']
        # wdnodes 包含行列标签
        region_nodes = raw['returndata']['wdnodes'][0]['nodes']  # 地区
        year_nodes   = raw['returndata']['wdnodes'][1]['nodes']  # 时间

        regions = {n['code']: n['cname'] for n in region_nodes}
        years   = {n['code']: int(n['cname']) for n in year_nodes}

        for node in nodes:
            code_parts = node['code'].split('.')   # e.g. "A060C01.110000.2024"
            if len(code_parts) < 3:
                continue
            reg_code  = code_parts[1]
            year_code = code_parts[2]
            value     = node['data'].get('strdata', '')
            if value and value not in ('', '--', 'null'):
                records.append({
                    'province': regions.get(reg_code, reg_code),
                    'year':     years.get(year_code, year_code),
                    'unit_cost': float(value),
                })
    except (KeyError, TypeError) as e:
        print(f"[NBS] JSON解析错误: {e}")

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.dropna(subset=['unit_cost'])
        df = df[df['unit_cost'] > 0]
    return df


# ─────────────────────────────────────────────
# 2. 全国公共资源交易平台 GGZY 住宅中标数据
# ─────────────────────────────────────────────

def fetch_ggzy_bid_results(keyword='住宅', max_pages=10, retries=3, delay=2):
    """
    从全国公共资源交易平台获取住宅工程中标数据
    包含：项目名称、中标金额、建设规模（面积）、省份、中标日期
    返回 DataFrame
    """
    base_url = "https://deal.ggzy.gov.cn/ds/deal/dealList_find.jsp"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Referer': 'https://deal.ggzy.gov.cn/',
    }

    all_results = []

    for page in range(1, max_pages + 1):
        payload = {
            'TIMEBEGIN_SHOW': '2020-01-01',
            'TIMEEND_SHOW': '2025-12-31',
            'DEAL_CLASSIFY': '01',   # 01=工程建设
            'KEYWORD': keyword,
            'currentPage': str(page),
            'pageSize': '20',
        }

        for attempt in range(retries):
            try:
                resp = requests.post(base_url, data=payload, headers=headers, timeout=30)
                resp.raise_for_status()
                results = _parse_ggzy_html(resp.text)
                all_results.extend(results)
                print(f"[GGZY] 第{page}页获取{len(results)}条")
                time.sleep(1.5)   # 礼貌性延迟
                break
            except Exception as e:
                print(f"[GGZY] 第{page}页第{attempt+1}次失败: {e}")
                if attempt < retries - 1:
                    time.sleep(delay * (2 ** attempt))

    if not all_results:
        print("[GGZY] 警告：未获取到数据，返回空DataFrame")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    return df


def _parse_ggzy_html(html_text):
    """解析GGZY列表页HTML，提取项目信息"""
    soup = BeautifulSoup(html_text, 'html.parser')
    results = []

    # GGZY的列表通常在table或特定div中，根据实际页面结构调整
    rows = soup.select('table.list-table tr') or soup.select('tr.list-item')

    for row in rows:
        cols = row.find_all('td')
        if len(cols) < 4:
            continue
        try:
            record = {
                'project_name': cols[0].get_text(strip=True),
                'province':     cols[1].get_text(strip=True),
                'bid_amount':   _parse_amount(cols[2].get_text(strip=True)),
                'bid_date':     cols[3].get_text(strip=True),
            }
            results.append(record)
        except (IndexError, ValueError):
            continue

    return results


def _parse_amount(text):
    """将金额字符串（如'1234.56万元'）转换为数值（元）"""
    text = text.replace(',', '').replace(' ', '')
    if '万' in text:
        num = float(''.join(c for c in text if c.isdigit() or c == '.'))
        return num * 10000
    elif text:
        try:
            return float(''.join(c for c in text if c.isdigit() or c == '.'))
        except ValueError:
            return np.nan
    return np.nan


# ─────────────────────────────────────────────
# 3. 材料价格指数
# ─────────────────────────────────────────────

# 螺纹钢年度均价（元/吨），数据来源：上海期货交易所公开数据/公开研报
STEEL_ANNUAL_PRICES = {
    2020: 3850,
    2021: 5200,
    2022: 4600,
    2023: 4100,
    2024: 3700,
    2025: 3600,
}

# 水泥年度均价（元/吨），数据来源：国家统计局水泥出厂价/水泥网
CEMENT_ANNUAL_PRICES = {
    2020: 430,
    2021: 520,
    2022: 480,
    2023: 390,
    2024: 350,
    2025: 340,
}


def calculate_price_index(annual_prices, base_year=2020):
    """
    以base_year为基期（=100）计算价格指数
    参数:
        annual_prices: dict, {年份: 价格}
        base_year: int, 基期年份
    返回:
        dict, {年份: 价格指数}
    """
    if base_year not in annual_prices:
        raise ValueError(f"基期年份 {base_year} 不在数据中")
    base_price = annual_prices[base_year]
    return {year: round(price / base_price * 100, 2)
            for year, price in annual_prices.items()}


def get_material_price_df():
    """
    构建材料价格指数DataFrame
    返回: DataFrame，columns=[year, steel_price_index, cement_price_index]
    """
    steel_idx  = calculate_price_index(STEEL_ANNUAL_PRICES,  base_year=2020)
    cement_idx = calculate_price_index(CEMENT_ANNUAL_PRICES, base_year=2020)

    years = sorted(set(steel_idx) & set(cement_idx))
    df = pd.DataFrame({
        'year':                years,
        'steel_price_index':  [steel_idx[y]  for y in years],
        'cement_price_index': [cement_idx[y] for y in years],
    })
    return df


# ─────────────────────────────────────────────
# 4. 辅助：生成合成备用数据（当真实数据不足时）
# ─────────────────────────────────────────────

def generate_fallback_data(n=350, random_state=42):
    """
    当真实数据采集量不足300条时的备用数据生成函数。
    基于真实工程造价统计规律构建，用于论文研究局限性章节说明。
    注意：此函数仅在数据量不足时使用，须在论文中注明。
    """
    rng = np.random.RandomState(random_state)

    provinces = ['北京', '上海', '广东', '浙江', '江苏', '四川',
                 '湖南', '湖北', '河南', '山东', '陕西', '重庆',
                 '福建', '安徽', '河北', '辽宁', '云南', '贵州']
    structure_types  = ['框架', '剪力墙', '框剪', '砌体']
    foundation_types = ['桩基础', '筏板基础', '独立基础', '条形基础']
    housing_types    = ['普通住宅', '高档住宅', '保障房', '别墅']
    decoration_map   = {'毛坯': 0, '简装': 1, '精装': 2, '豪装': 3}

    # 省份基准造价（元/㎡），参考真实统计数据
    province_base = {
        '北京': 4500, '上海': 4800, '广东': 4200, '浙江': 4000,
        '江苏': 3800, '四川': 3200, '湖南': 3000, '湖北': 3100,
        '河南': 2900, '山东': 3100, '陕西': 3000, '重庆': 3300,
        '福建': 3500, '安徽': 2800, '河北': 2900, '辽宁': 3000,
        '云南': 2700, '贵州': 2600,
    }

    material_df = get_material_price_df()
    price_dict = dict(zip(material_df['year'],
                          zip(material_df['steel_price_index'],
                              material_df['cement_price_index'])))

    records = []
    for _ in range(n):
        province       = rng.choice(provinces)
        year           = rng.randint(2020, 2026)
        above_floors   = rng.randint(3, 34)
        under_floors   = rng.randint(0, 3)
        total_area     = rng.uniform(5000, 80000)
        struct         = rng.choice(structure_types)
        foundation     = rng.choice(foundation_types)
        seismic        = rng.choice([6, 7, 8])
        is_prefab      = int(rng.random() < 0.20)
        dec_level      = rng.choice(list(decoration_map.values()))
        housing_type   = rng.choice(housing_types)

        # 合成造价（基准 + 影响因子 + 随机扰动）
        base = province_base[province]
        steel_idx, cement_idx = price_dict.get(year, (100, 100))

        # 层数影响：高层造价偏高
        floor_factor = 1 + (above_floors - 10) * 0.005 if above_floors > 10 else 1.0
        # 装修标准影响
        dec_factor = 1 + dec_level * 0.08
        # 材料价格影响
        mat_factor = (steel_idx + cement_idx) / 200
        # 年份趋势
        year_trend = 1 + (year - 2020) * 0.02

        unit_cost = (base * floor_factor * dec_factor *
                     mat_factor * year_trend *
                     rng.uniform(0.90, 1.10))   # ±10%扰动

        records.append({
            'province':          province,
            'completion_year':   year,
            'total_area':        round(total_area, 1),
            'above_floors':      above_floors,
            'under_floors':      under_floors,
            'structure_type':    struct,
            'foundation_type':   foundation,
            'seismic_intensity': seismic,
            'is_prefab':         is_prefab,
            'decoration_level':  dec_level,
            'housing_type':      housing_type,
            'steel_price_index': steel_idx,
            'cement_price_index': cement_idx,
            'unit_cost':         round(unit_cost, 2),
            'data_source':       'synthetic_fallback',
        })

    df = pd.DataFrame(records)
    print(f"[备用数据] 已生成 {len(df)} 条合成数据（须在论文局限性章节注明）")
    return df
