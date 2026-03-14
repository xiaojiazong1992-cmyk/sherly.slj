# -*- coding: utf-8 -*-
"""
data_scraper.py
真实数据爬取模块
数据源优先级：
  A. 国家统计局 NBS easyquery API（分省年度竣工造价）
  B. 全国公共资源交易平台 GGZY（住宅工程中标记录）
  C. 住房城乡建设部造价信息（补充）
当所有在线数据源不可达时（网络受限环境），自动回退到统计规律构建的备用数据，
并在日志中明确标注，使用者应在论文"研究局限性"章节注明。
"""

import time
import json
import logging
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 通用工具
# ─────────────────────────────────────────────────────────────────────────────

def _get(url, params=None, headers=None, timeout=20, retries=3, delay=2):
    """带指数退避重试的GET请求"""
    _headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/120.0.0.0 Safari/537.36'),
        'Accept': 'application/json, text/html, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    }
    if headers:
        _headers.update(headers)
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=_headers, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            logger.warning(f"GET {url} 第{attempt+1}次失败: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))
    return None


def _post(url, data=None, headers=None, timeout=20, retries=3, delay=2):
    """带指数退避重试的POST请求"""
    _headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/120.0.0.0 Safari/537.36'),
        'Content-Type': 'application/x-www-form-urlencoded',
        'Referer': 'https://deal.ggzy.gov.cn/',
    }
    if headers:
        _headers.update(headers)
    for attempt in range(retries):
        try:
            r = requests.post(url, data=data, headers=_headers, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            logger.warning(f"POST {url} 第{attempt+1}次失败: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# A. 国家统计局 NBS
#    指标 A060C01：各地区房屋竣工造价（元/平方米）
#    来源：https://data.stats.gov.cn/easyquery.htm
# ─────────────────────────────────────────────────────────────────────────────

NBS_URL = 'https://data.stats.gov.cn/easyquery.htm'

# 省份代码→名称映射（NBS地区代码）
NBS_REGION_MAP = {
    '110000': '北京', '120000': '天津', '130000': '河北', '140000': '山西',
    '150000': '内蒙古', '210000': '辽宁', '220000': '吉林', '230000': '黑龙江',
    '310000': '上海', '320000': '江苏', '330000': '浙江', '340000': '安徽',
    '350000': '福建', '360000': '江西', '370000': '山东', '410000': '河南',
    '420000': '湖北', '430000': '湖南', '440000': '广东', '450000': '广西',
    '460000': '海南', '500000': '重庆', '510000': '四川', '520000': '贵州',
    '530000': '云南', '540000': '西藏', '610000': '陕西', '620000': '甘肃',
    '630000': '青海', '640000': '宁夏', '650000': '新疆',
}


def fetch_nbs_construction_cost(years='LAST6', retries=3, delay=3):
    """
    从国家统计局 easyquery API 获取各省房屋竣工造价
    指标代码 A060C01：各地区房屋竣工造价（元/平方米）
    覆盖范围：31省 × 近6年 ≈ 186条真实数据

    返回 DataFrame: columns=[province, year, unit_cost, data_source]
    """
    params = {
        'm': 'QueryData',
        'dbcode': 'fsnd',
        'rowcode': 'reg',
        'colcode': 'sj',
        'wds': '[]',
        'dfwds': json.dumps([
            {'wdcode': 'zb',  'valuecode': 'A060C01'},
            {'wdcode': 'sj',  'valuecode': years},
        ]),
    }
    headers = {
        'Referer': NBS_URL,
        'Accept': 'application/json',
    }

    logger.info("正在请求国家统计局 NBS API...")
    resp = _get(NBS_URL, params=params, headers=headers,
                timeout=25, retries=retries, delay=delay)
    if resp is None:
        logger.error("NBS API 无法访问（网络受限或被限流）")
        return pd.DataFrame(columns=['province', 'year', 'unit_cost', 'data_source'])

    try:
        raw = resp.json()
    except Exception as e:
        logger.error(f"NBS 响应非JSON: {e}")
        return pd.DataFrame(columns=['province', 'year', 'unit_cost', 'data_source'])

    return _parse_nbs_json(raw)


def _parse_nbs_json(raw):
    """解析NBS easyquery返回的JSON，提取省份/年份/造价"""
    records = []
    try:
        returndata = raw.get('returndata', {})
        datanodes  = returndata.get('datanodes', [])
        wdnodes    = returndata.get('wdnodes', [])

        # 构建地区/年份 code→name 映射
        region_map = {}
        year_map   = {}
        for wdn in wdnodes:
            wdcode = wdn.get('wdcode', '')
            nodes  = wdn.get('nodes', [])
            if wdcode == 'reg':
                region_map = {n['code']: n['cname'] for n in nodes}
            elif wdcode == 'sj':
                for n in nodes:
                    # cname 格式可能是 '2024年' 或 '2024'
                    yr_str = n['cname'].replace('年', '').strip()
                    try:
                        year_map[n['code']] = int(yr_str)
                    except ValueError:
                        pass

        # 如果wdnodes里没有地区，用内置映射兜底
        if not region_map:
            region_map = NBS_REGION_MAP

        for node in datanodes:
            code = node.get('code', '')
            # code格式: "A060C01.130000.2024"
            parts = code.split('.')
            if len(parts) < 3:
                continue
            reg_code  = parts[1]
            year_code = parts[2]
            val_str   = node.get('data', {}).get('strdata', '')

            if val_str and val_str not in ('', '--', 'null', 'None'):
                try:
                    province = region_map.get(reg_code, reg_code)
                    year     = year_map.get(year_code)
                    if year is None:
                        # 尝试直接解析年份code
                        year = int(year_code[:4]) if len(year_code) >= 4 else None
                    if year is None:
                        continue
                    records.append({
                        'province':    province,
                        'year':        year,
                        'unit_cost':   float(val_str),
                        'data_source': 'NBS',
                    })
                except (ValueError, TypeError):
                    continue

    except Exception as e:
        logger.error(f"NBS JSON解析失败: {e}")

    df = pd.DataFrame(records)
    if not df.empty:
        df = df[df['unit_cost'] > 0].dropna(subset=['unit_cost', 'year'])
        logger.info(f"NBS 解析成功: {len(df)} 条（{df['year'].min()}-{df['year'].max()}年，"
                    f"{df['province'].nunique()}个省份）")
    else:
        logger.warning("NBS 未解析到有效数据")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# B. 全国公共资源交易平台 GGZY
#    住宅工程中标公告（包含建筑面积、中标价格）
#    来源：https://deal.ggzy.gov.cn/
# ─────────────────────────────────────────────────────────────────────────────

GGZY_URL = 'https://deal.ggzy.gov.cn/ds/deal/dealList_find.jsp'


def fetch_ggzy_bid_results(keyword='住宅', max_pages=15, delay=2.0):
    """
    从全国公共资源交易平台爬取住宅工程中标数据（2020-2025）
    包含：项目名称、省份、中标金额、建设面积、日期

    注意：该平台有反爬措施，爬取过于频繁会被封IP
    建议：max_pages ≤ 20，delay ≥ 2.0秒
    """
    all_records = []

    for page in range(1, max_pages + 1):
        payload = {
            'TIMEBEGIN_SHOW': '2020-01-01',
            'TIMEEND_SHOW':   '2025-12-31',
            'DEAL_CLASSIFY':  '01',       # 工程建设
            'DEAL_STAGE':     '0400',     # 中标/成交阶段
            'KEYWORD':        keyword,
            'currentPage':    str(page),
            'pageSize':       '20',
        }
        resp = _post(GGZY_URL, data=payload, retries=3, delay=delay)
        if resp is None:
            logger.warning(f"GGZY 第{page}页获取失败，停止分页")
            break

        records = _parse_ggzy_html(resp.text)
        if not records:
            logger.info(f"GGZY 第{page}页无数据，停止")
            break

        all_records.extend(records)
        logger.info(f"GGZY 第{page}页: {len(records)}条，累计{len(all_records)}条")
        time.sleep(delay)   # 礼貌延迟

    if not all_records:
        logger.warning("GGZY 未获取到任何数据")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df['data_source'] = 'GGZY'
    return df


def _parse_ggzy_html(html_text):
    """解析GGZY列表页，提取项目基本信息"""
    soup = BeautifulSoup(html_text, 'html.parser')
    records = []

    # GGZY页面结构可能随改版变化，此处覆盖常见几种选择器
    rows = (soup.select('table.table-list tbody tr') or
            soup.select('table.list-table tr') or
            soup.select('tr.list-item') or
            soup.select('div.deal-item'))

    for row in rows:
        cols = row.find_all(['td', 'div'])
        if len(cols) < 4:
            continue
        try:
            records.append({
                'project_name': cols[0].get_text(strip=True),
                'province':     cols[1].get_text(strip=True)[:3],  # 取前3字
                'bid_amount':   _parse_currency(cols[2].get_text(strip=True)),
                'bid_date':     cols[3].get_text(strip=True),
            })
        except (IndexError, ValueError):
            continue
    return records


def _parse_currency(text):
    """解析金额字符串 → 元（数值）"""
    text = text.replace(',', '').replace(' ', '')
    try:
        if '亿' in text:
            return float(''.join(c for c in text.split('亿')[0] if c.isdigit() or c == '.')) * 1e8
        elif '万' in text:
            return float(''.join(c for c in text.split('万')[0] if c.isdigit() or c == '.')) * 1e4
        else:
            return float(''.join(c for c in text if c.isdigit() or c == '.')) if text else np.nan
    except ValueError:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# C. 将NBS省级数据扩展为项目级记录
#    NBS只有省级年度均值，需要扩展为含工程技术特征的行级记录
#    技术特征基于全国住宅工程统计分布生成（参考住建部统计年鉴）
# ─────────────────────────────────────────────────────────────────────────────

def expand_nbs_to_projects(df_nbs, projects_per_record=4, random_state=42):
    """
    将NBS省级年度造价扩展为项目级数据集

    数据增强依据：
    - 结构类型分布：剪力墙45%、框架25%、框剪20%、砌体10%（住建部2023年鉴）
    - 装修标准分布：毛坯30%、简装30%、精装30%、豪装10%（市场调研）
    - 省内项目差异：以省均值为中心，±15%随机扰动（参考各省竣工数据分布）

    论文注明：技术特征由统计分布生成，造价锚定NBS省级真实均值
    """
    if df_nbs.empty:
        return pd.DataFrame()

    rng = np.random.RandomState(random_state)

    STRUCT_TYPES  = ['框架', '剪力墙', '框剪', '砌体']
    STRUCT_PROBS  = [0.25, 0.45, 0.20, 0.10]
    FOUND_TYPES   = ['桩基础', '筏板基础', '独立基础', '条形基础']
    HOUSING_TYPES = ['普通住宅', '高档住宅', '保障房', '别墅']
    HOUSING_PROBS = [0.50, 0.20, 0.25, 0.05]
    SEISMIC_VALS  = [6, 7, 8]
    SEISMIC_PROBS = [0.30, 0.50, 0.20]

    records = []
    for _, row in df_nbs.iterrows():
        for _ in range(projects_per_record):
            above_floors = int(rng.randint(3, 34))
            total_area   = float(rng.uniform(5000, 80000))
            # 以NBS真实省级均价为锚点，±15%扰动模拟项目间差异
            unit_cost    = float(row['unit_cost'] * rng.uniform(0.85, 1.15))

            records.append({
                'province':          row['province'],
                'completion_year':   int(row['year']),
                'total_area':        round(total_area, 1),
                'above_floors':      above_floors,
                'under_floors':      int(rng.randint(0, 3)),
                'structure_type':    rng.choice(STRUCT_TYPES, p=STRUCT_PROBS),
                'foundation_type':   rng.choice(FOUND_TYPES),
                'seismic_intensity': int(rng.choice(SEISMIC_VALS, p=SEISMIC_PROBS)),
                'is_prefab':         int(rng.random() < 0.18),
                'decoration_level':  int(rng.choice([0, 1, 2, 3], p=[0.30, 0.30, 0.30, 0.10])),
                'housing_type':      rng.choice(HOUSING_TYPES, p=HOUSING_PROBS),
                'unit_cost':         round(unit_cost, 2),
                'data_source':       'NBS_real_expanded',  # 造价为NBS真实值，技术特征为统计扩展
            })

    df = pd.DataFrame(records)
    logger.info(f"NBS扩展完成: {len(df_nbs)}条省级数据 → {len(df)}条项目级数据")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# D. 材料价格指数
#    来源：上海期货交易所螺纹钢主力合约年均价 + 中国水泥协会/统计局水泥出厂价
#    数据截至2025年（2025年为估算值，以2024年实际数据为基础）
# ─────────────────────────────────────────────────────────────────────────────

# 螺纹钢年度均价（元/吨）— 来源：SHFE螺纹钢主力合约年均价（公开数据）
STEEL_ANNUAL_PRICES = {
    2020: 3850,   # SHFE rb主力合约2020年均价
    2021: 5200,   # 钢价高峰（碳达峰政策+基建需求）
    2022: 4600,   # 回落（需求下滑）
    2023: 4100,   # 持续下行
    2024: 3700,   # 低位震荡
    2025: 3600,   # 预估（基于2024Q4走势）
}

# 水泥年度均价（元/吨）— 来源：中国水泥协会/国家统计局水泥出厂价格指数
CEMENT_ANNUAL_PRICES = {
    2020: 430,    # 疫情后恢复
    2021: 520,    # 大幅上涨（错峰限产+需求旺盛）
    2022: 480,    # 地产下行带动回调
    2023: 390,    # 明显下滑
    2024: 350,    # 历史低位
    2025: 340,    # 预估（产能过剩延续）
}


def calculate_price_index(annual_prices, base_year=2020):
    """
    以base_year为基期（=100）计算价格指数
    参数:
        annual_prices: dict {年份: 价格}
        base_year: 基期年份
    返回: dict {年份: 价格指数}
    """
    if base_year not in annual_prices:
        raise ValueError(f"基期年份{base_year}不在数据中，可用年份: {list(annual_prices)}")
    base = annual_prices[base_year]
    return {yr: round(price / base * 100, 2) for yr, price in annual_prices.items()}


def get_material_price_df():
    """
    构建材料价格指数DataFrame（以2020年为基期=100）
    返回: DataFrame columns=[year, steel_price_index, cement_price_index,
                               steel_price_yuan, cement_price_yuan]
    """
    steel_idx  = calculate_price_index(STEEL_ANNUAL_PRICES,  base_year=2020)
    cement_idx = calculate_price_index(CEMENT_ANNUAL_PRICES, base_year=2020)
    years = sorted(set(steel_idx) & set(cement_idx))

    df = pd.DataFrame({
        'year':                years,
        'steel_price_index':  [steel_idx[y]  for y in years],
        'cement_price_index': [cement_idx[y] for y in years],
        'steel_price_yuan':   [STEEL_ANNUAL_PRICES[y]  for y in years],
        'cement_price_yuan':  [CEMENT_ANNUAL_PRICES[y] for y in years],
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# E. 主入口：自动尝试真实数据，失败则回退合成数据
# ─────────────────────────────────────────────────────────────────────────────

def collect_data(min_samples=300, random_state=42):
    """
    数据采集主函数，按优先级尝试：
    1. NBS API → 省级真实造价 → 扩展为项目级
    2. GGZY API → 项目级中标数据
    3. 合成备用数据（须在论文局限性注明）

    返回: (df_main, data_quality_report)
    """
    report = {}
    all_dfs = []
    material_df = get_material_price_df()

    # ── Step A: NBS ───────────────────────────────────────────────────────────
    df_nbs = fetch_nbs_construction_cost()
    report['nbs_raw_rows'] = len(df_nbs)

    if not df_nbs.empty:
        df_expanded = expand_nbs_to_projects(df_nbs, projects_per_record=4,
                                              random_state=random_state)
        report['nbs_expanded_rows'] = len(df_expanded)
        all_dfs.append(df_expanded)
        logger.info(f"✓ NBS数据：{len(df_nbs)}条省级 → 扩展{len(df_expanded)}条项目级")
    else:
        report['nbs_expanded_rows'] = 0
        logger.warning("✗ NBS数据获取失败")

    # ── Step B: GGZY ─────────────────────────────────────────────────────────
    df_ggzy = fetch_ggzy_bid_results(keyword='住宅', max_pages=10)
    report['ggzy_rows'] = len(df_ggzy)
    if not df_ggzy.empty:
        all_dfs.append(df_ggzy)
        logger.info(f"✓ GGZY数据：{len(df_ggzy)}条中标记录")

    # ── Step C: 合并 ─────────────────────────────────────────────────────────
    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
    else:
        df = pd.DataFrame()

    # ── Step D: 数量不足则用备用数据补充 ────────────────────────────────────
    if len(df) < min_samples:
        shortfall = max(min_samples, min_samples - len(df))
        logger.warning(f"⚠ 数据量不足 ({len(df)} < {min_samples})，"
                       f"补充{shortfall}条统计规律合成数据")
        logger.warning("  → 须在论文'研究局限性'章节注明使用了合成数据")
        df_fb = generate_fallback_data(n=shortfall, random_state=random_state)
        df = pd.concat([df, df_fb], ignore_index=True) if not df.empty else df_fb

    report['final_rows']  = len(df)
    report['uses_real_nbs'] = report['nbs_raw_rows'] > 0

    # ── Step E: 关联材料价格指数 ─────────────────────────────────────────────
    df = _merge_material_prices(df, material_df)
    return df, report


def _merge_material_prices(df, material_df):
    """将材料价格指数按竣工年份关联到主数据"""
    year_col = 'completion_year' if 'completion_year' in df.columns else 'year'
    if year_col not in df.columns:
        return df

    mat = material_df.rename(columns={'year': year_col})

    # 合并（如果已有这些列则覆盖）
    drop_cols = [c for c in ['steel_price_index', 'cement_price_index'] if c in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')
    df = df.merge(mat[[year_col, 'steel_price_index', 'cement_price_index']],
                  on=year_col, how='left')
    return df


# ─────────────────────────────────────────────────────────────────────────────
# F. 备用合成数据（仅当所有真实数据源不可达时使用）
# ─────────────────────────────────────────────────────────────────────────────

def generate_fallback_data(n=350, random_state=42):
    """
    基于统计规律的合成备用数据
    ⚠ 仅在网络受限、无法访问NBS/GGZY时使用
    ⚠ 使用时须在论文"研究局限性"章节明确注明

    合成依据：
    - 省份基准造价参考：住建部《2023年城市建设统计年鉴》
    - 造价影响因子参考：各省工程造价信息发布价
    - 材料价格：SHFE螺纹钢主力合约年均价 + 中国水泥协会
    """
    rng = np.random.RandomState(random_state)

    PROVINCES = ['北京', '上海', '广东', '浙江', '江苏', '四川',
                 '湖南', '湖北', '河南', '山东', '陕西', '重庆',
                 '福建', '安徽', '河北', '辽宁', '云南', '贵州']

    # 省份基准造价（元/㎡，2020年基准，参考住建部统计年鉴）
    PROVINCE_BASE = {
        '北京': 4500, '上海': 4800, '广东': 4200, '浙江': 4000,
        '江苏': 3800, '四川': 3200, '湖南': 3000, '湖北': 3100,
        '河南': 2900, '山东': 3100, '陕西': 3000, '重庆': 3300,
        '福建': 3500, '安徽': 2800, '河北': 2900, '辽宁': 3000,
        '云南': 2700, '贵州': 2600,
    }

    STRUCT_TYPES  = ['框架', '剪力墙', '框剪', '砌体']
    STRUCT_PROBS  = [0.25, 0.45, 0.20, 0.10]
    FOUND_TYPES   = ['桩基础', '筏板基础', '独立基础', '条形基础']
    HOUSING_TYPES = ['普通住宅', '高档住宅', '保障房', '别墅']
    HOUSING_PROBS = [0.50, 0.20, 0.25, 0.05]

    material_df = get_material_price_df()
    price_idx   = {row['year']: (row['steel_price_index'], row['cement_price_index'])
                   for _, row in material_df.iterrows()}

    records = []
    for _ in range(n):
        province     = rng.choice(PROVINCES)
        year         = int(rng.randint(2020, 2026))
        above_floors = int(rng.randint(3, 34))
        total_area   = float(rng.uniform(5000, 80000))
        steel_idx, cement_idx = price_idx.get(year, (100.0, 100.0))

        base         = PROVINCE_BASE[province]
        floor_factor = 1 + max(0, above_floors - 10) * 0.005
        dec_level    = int(rng.choice([0, 1, 2, 3], p=[0.30, 0.30, 0.30, 0.10]))
        dec_factor   = 1 + dec_level * 0.08
        mat_factor   = (steel_idx + cement_idx) / 200
        year_trend   = 1 + (year - 2020) * 0.02
        unit_cost    = (base * floor_factor * dec_factor *
                        mat_factor * year_trend * rng.uniform(0.90, 1.10))

        records.append({
            'province':          province,
            'completion_year':   year,
            'total_area':        round(total_area, 1),
            'above_floors':      above_floors,
            'under_floors':      int(rng.randint(0, 3)),
            'structure_type':    rng.choice(STRUCT_TYPES, p=STRUCT_PROBS),
            'foundation_type':   rng.choice(FOUND_TYPES),
            'seismic_intensity': int(rng.choice([6, 7, 8], p=[0.30, 0.50, 0.20])),
            'is_prefab':         int(rng.random() < 0.18),
            'decoration_level':  dec_level,
            'housing_type':      rng.choice(HOUSING_TYPES, p=HOUSING_PROBS),
            'steel_price_index': steel_idx,
            'cement_price_index': cement_idx,
            'unit_cost':         round(unit_cost, 2),
            'data_source':       'synthetic_fallback',
        })

    df = pd.DataFrame(records)
    logger.warning(f"[备用数据] 已生成{len(df)}条合成数据（须在论文局限性章节注明）")
    return df
